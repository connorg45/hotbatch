use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KvHandle {
    slot: usize,
    capacity_tokens: usize,
    generation: u64,
}

impl KvHandle {
    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }
}

pub trait KvCache {
    fn has_room_for(&self, prompt_len: usize, max_new: usize) -> bool;
    fn allocate(&mut self, prompt_len: usize, max_new: usize) -> Result<KvHandle>;
    fn free(&mut self, handle: KvHandle);
    fn write(&mut self, handle: &KvHandle, layer: usize, k: &Tensor, v: &Tensor) -> Result<()>;
    fn read(&self, handle: &KvHandle, layer: usize) -> Result<(Tensor, Tensor)>;
}

#[derive(Debug)]
pub struct SlabKvCache {
    max_seqs: usize,
    max_seq_len: usize,
    num_layers: usize,
    n_heads: usize,
    head_dim: usize,
    free_slots: VecDeque<usize>,
    allocated: HashMap<usize, KvHandle>,
    next_generations: Vec<u64>,
    layers: HashMap<usize, Vec<Option<(Tensor, Tensor)>>>,
    device: Device,
}

impl SlabKvCache {
    pub fn new(
        max_seqs: usize,
        max_seq_len: usize,
        num_layers: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        let free_slots = (0..max_seqs).collect();
        Self {
            max_seqs,
            max_seq_len,
            num_layers,
            n_heads,
            head_dim,
            free_slots,
            allocated: HashMap::new(),
            next_generations: vec![0; max_seqs],
            layers: HashMap::new(),
            device: Device::Cpu,
        }
    }

    pub fn allocated_slots(&self) -> usize {
        self.allocated.len()
    }

    pub fn max_sequences(&self) -> usize {
        self.max_seqs
    }

    pub fn max_sequence_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn capacity_shape(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.max_seqs,
            self.num_layers,
            2,
            self.max_seq_len,
            self.n_heads,
            self.head_dim,
        )
    }

    fn is_current_handle(&self, handle: &KvHandle) -> bool {
        self.allocated.get(&handle.slot) == Some(handle)
    }
}

impl KvCache for SlabKvCache {
    fn has_room_for(&self, prompt_len: usize, max_new: usize) -> bool {
        prompt_len
            .checked_add(max_new)
            .is_some_and(|tokens| tokens <= self.max_seq_len)
            && !self.free_slots.is_empty()
    }

    fn allocate(&mut self, prompt_len: usize, max_new: usize) -> Result<KvHandle> {
        if !self.has_room_for(prompt_len, max_new) {
            return Err(anyhow!(
                "kv cache full or requested sequence too long: prompt_len={prompt_len}, max_new={max_new}, max_seq_len={}",
                self.max_seq_len
            ));
        }
        let Some(slot) = self.free_slots.pop_front() else {
            return Err(anyhow!("kv cache has no free slots"));
        };
        let generation = self.next_generations[slot].wrapping_add(1);
        self.next_generations[slot] = generation;
        let handle = KvHandle {
            slot,
            capacity_tokens: prompt_len + max_new,
            generation,
        };
        self.allocated.insert(slot, handle.clone());
        self.layers.insert(slot, vec![None; self.num_layers]);
        Ok(handle)
    }

    fn free(&mut self, handle: KvHandle) {
        if self.is_current_handle(&handle) {
            self.allocated.remove(&handle.slot);
            self.layers.remove(&handle.slot);
            self.free_slots.push_back(handle.slot);
        }
    }

    fn write(&mut self, handle: &KvHandle, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        if !self.is_current_handle(handle) {
            return Err(anyhow!(
                "attempted to write kv with stale or unallocated handle for slot {}",
                handle.slot
            ));
        }
        if layer >= self.num_layers {
            return Err(anyhow!("kv layer {layer} out of range {}", self.num_layers));
        }
        let Some(layers) = self.layers.get_mut(&handle.slot) else {
            return Err(anyhow!("kv slot {} missing layer storage", handle.slot));
        };
        let k_dims = k.dims();
        let v_dims = v.dims();
        if k_dims.len() != 3 || v_dims.len() != 3 {
            return Err(anyhow!(
                "kv tensors must be [heads, tokens, head_dim], got k={k_dims:?}, v={v_dims:?}"
            ));
        }
        if k_dims != v_dims {
            return Err(anyhow!(
                "kv key/value shape mismatch: k={k_dims:?}, v={v_dims:?}"
            ));
        }
        if k_dims[0] != self.n_heads || v_dims[0] != self.n_heads {
            return Err(anyhow!(
                "kv head mismatch: expected {}, got k={}, v={}",
                self.n_heads,
                k_dims[0],
                v_dims[0]
            ));
        }
        if k_dims[2] != self.head_dim || v_dims[2] != self.head_dim {
            return Err(anyhow!(
                "kv head_dim mismatch: expected {}, got k={}, v={}",
                self.head_dim,
                k_dims[2],
                v_dims[2]
            ));
        }
        if k_dims[1] > handle.capacity_tokens || v_dims[1] > handle.capacity_tokens {
            return Err(anyhow!(
                "kv token capacity exceeded: capacity={}, k_tokens={}, v_tokens={}",
                handle.capacity_tokens,
                k_dims[1],
                v_dims[1]
            ));
        }
        layers[layer] = Some((k.clone(), v.clone()));
        Ok(())
    }

    fn read(&self, handle: &KvHandle, layer: usize) -> Result<(Tensor, Tensor)> {
        if !self.is_current_handle(handle) {
            return Err(anyhow!(
                "attempted to read kv with stale or unallocated handle for slot {}",
                handle.slot
            ));
        }
        if layer >= self.num_layers {
            return Err(anyhow!("kv layer {layer} out of range {}", self.num_layers));
        }
        let Some(layers) = self.layers.get(&handle.slot) else {
            return Err(anyhow!("kv slot {} missing layer storage", handle.slot));
        };
        match layers.get(layer).and_then(|entry| entry.as_ref()) {
            Some((k, v)) => Ok((k.clone(), v.clone())),
            None => {
                let shape = (self.n_heads, 0, self.head_dim);
                let k = Tensor::zeros(shape, DType::F32, &self.device)?;
                let v = Tensor::zeros(shape, DType::F32, &self.device)?;
                Ok((k, v))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_capacity_and_accepts_exact_token_limit() {
        let mut cache = SlabKvCache::new(2, 8, 3, 4, 5);

        assert_eq!(cache.capacity_shape(), (2, 3, 2, 8, 4, 5));
        assert_eq!(cache.max_sequences(), 2);
        assert_eq!(cache.max_sequence_len(), 8);
        assert!(cache.has_room_for(3, 5));
        assert!(!cache.has_room_for(4, 5));
        assert!(!cache.has_room_for(usize::MAX, 1));

        let handle = cache.allocate(3, 5).expect("exact capacity should fit");
        assert_eq!(handle.capacity_tokens(), 8);
    }

    #[test]
    fn allocation_exhaustion_release_and_reuse_are_consistent() {
        let mut cache = SlabKvCache::new(2, 8, 1, 1, 1);
        let first = cache.allocate(2, 2).expect("first slot");
        let second = cache.allocate(1, 1).expect("second slot");

        assert_eq!(cache.allocated_slots(), 2);
        assert!(cache.allocate(1, 1).is_err());

        let first_slot = first.slot();
        cache.free(first.clone());
        assert_eq!(cache.allocated_slots(), 1);

        let reused = cache
            .allocate(4, 4)
            .expect("released slot should be reused");
        assert_eq!(reused.slot(), first_slot);
        assert_ne!(reused, first);
        assert_eq!(cache.allocated_slots(), 2);

        cache.free(first);
        assert_eq!(cache.allocated_slots(), 2, "stale free must be ignored");
        cache.free(reused);
        cache.free(second);
        assert_eq!(cache.allocated_slots(), 0);
    }

    #[test]
    fn writes_enforce_handle_and_token_capacity() {
        let mut cache = SlabKvCache::new(1, 4, 1, 2, 3);
        let handle = cache.allocate(2, 2).expect("slot");
        let exact = Tensor::zeros((2, 4, 3), DType::F32, &Device::Cpu).expect("tensor");
        cache
            .write(&handle, 0, &exact, &exact)
            .expect("exact token capacity should fit");

        let oversized = Tensor::zeros((2, 5, 3), DType::F32, &Device::Cpu).expect("tensor");
        assert!(cache.write(&handle, 0, &oversized, &oversized).is_err());

        cache.free(handle.clone());
        assert!(cache.read(&handle, 0).is_err());
        assert!(cache.write(&handle, 0, &exact, &exact).is_err());
    }

    #[test]
    fn writes_reject_mismatched_key_and_value_shapes() {
        let mut cache = SlabKvCache::new(1, 4, 1, 2, 3);
        let handle = cache.allocate(2, 2).expect("slot");
        let key = Tensor::zeros((2, 3, 3), DType::F32, &Device::Cpu).expect("key");
        let value = Tensor::zeros((2, 2, 3), DType::F32, &Device::Cpu).expect("value");

        assert!(cache.write(&handle, 0, &key, &value).is_err());
    }
}
