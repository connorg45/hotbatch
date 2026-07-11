use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KvHandle {
    slot: usize,
    capacity_tokens: usize,
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
    allocated: HashSet<usize>,
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
            allocated: HashSet::new(),
            layers: HashMap::new(),
            device: Device::Cpu,
        }
    }

    pub fn allocated_slots(&self) -> usize {
        self.allocated.len()
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
}

impl KvCache for SlabKvCache {
    fn has_room_for(&self, prompt_len: usize, max_new: usize) -> bool {
        prompt_len.saturating_add(max_new) <= self.max_seq_len && !self.free_slots.is_empty()
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
        self.allocated.insert(slot);
        self.layers.insert(slot, vec![None; self.num_layers]);
        Ok(KvHandle {
            slot,
            capacity_tokens: prompt_len.saturating_add(max_new),
        })
    }

    fn free(&mut self, handle: KvHandle) {
        if self.allocated.remove(&handle.slot) {
            self.layers.remove(&handle.slot);
            self.free_slots.push_back(handle.slot);
        }
    }

    fn write(&mut self, handle: &KvHandle, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        if !self.allocated.contains(&handle.slot) {
            return Err(anyhow!(
                "attempted to write kv for unallocated slot {}",
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
        if !self.allocated.contains(&handle.slot) {
            return Err(anyhow!(
                "attempted to read kv for unallocated slot {}",
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
