//! Byte geometry only; neither arithmetic nor physical admission.
use super::*;

pub(in crate::resident_leases) const WINDOW: usize = 1024 * 1024;
pub(in crate::resident_leases) const CONTROL: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::resident_leases) struct Geometry {
    rows: usize,
    row_bytes: usize,
    bytes: usize,
}
impl Geometry {
    pub(in crate::resident_leases) fn new(rows: usize, columns: usize) -> Result<Self> {
        let row_bytes = columns
            .checked_mul(2)
            .filter(|n| *n != 0 && *n <= WINDOW && n.is_multiple_of(16))
            .ok_or_else(|| Error::Limit("BF16 matrix aligned complete row".into()))?;
        let bytes = rows
            .checked_mul(row_bytes)
            .filter(|_| rows != 0 && rows - 1 <= u32::MAX as usize)
            .ok_or_else(|| Error::Limit("BF16 matrix row/byte overflow".into()))?;
        Ok(Self {
            rows,
            row_bytes,
            bytes,
        })
    }
    pub(in crate::resident_leases) fn rows(self) -> usize {
        self.rows
    }
    pub(in crate::resident_leases) fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub(in crate::resident_leases) fn bytes(self) -> usize {
        self.bytes
    }
    pub(in crate::resident_leases) fn windows(self) -> usize {
        self.rows.div_ceil(WINDOW / self.row_bytes)
    }
    pub(in crate::resident_leases) fn output_bytes(self, ids: &[u32]) -> Result<usize> {
        ids.len()
            .checked_mul(self.row_bytes)
            .filter(|n| *n != 0 && *n <= WINDOW && ids.iter().all(|id| (*id as usize) < self.rows))
            .ok_or_else(|| Error::Limit("BF16 matrix bounded actual row IDs".into()))
    }
    pub(in crate::resident_leases) fn portable_limits(self, maximum: u64) -> Result<()> {
        // Monolithic storage only. Larger source matrices need a separately
        // proved paged lowering; advertised VRAM is not a storage-binding limit.
        if self.bytes as u64 > maximum || self.bytes / 4 > u32::MAX as usize {
            return Err(Error::Unsupported(
                "BF16 monolithic matrix page/binding/u32 limit".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::resident_leases) fn region(self, first: usize) -> Result<Option<Region>> {
        if first > self.rows {
            return Err(Error::State("BF16 upload coverage overflow".into()));
        }
        if first == self.rows {
            return Ok(None);
        }
        let end = first
            .checked_add((WINDOW / self.row_bytes).min(self.rows - first))
            .ok_or_else(|| Error::Limit("BF16 window row overflow".into()))?;
        Ok(Some(Region {
            first,
            end,
            offset: first * self.row_bytes,
            bytes: (end - first) * self.row_bytes,
        }))
    }
    #[cfg(feature = "vllm-cuda-qualification")]
    pub(in crate::resident_leases) fn cuda(self) -> Result<cubecl::cuda::Bf16MatrixLayout> {
        cubecl::cuda::Bf16MatrixLayout::new(self.rows, self.row_bytes / 2)
            .map_err(|e| Error::State(format!("BF16 CUDA layout: {e}")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::resident_leases) struct Region {
    pub(in crate::resident_leases) first: usize,
    pub(in crate::resident_leases) end: usize,
    pub(in crate::resident_leases) offset: usize,
    pub(in crate::resident_leases) bytes: usize,
}
pub(in crate::resident_leases) struct Coverage {
    geometry: Geometry,
    completed: usize,
    pending: Option<Region>,
}
impl Coverage {
    pub(in crate::resident_leases) fn new(geometry: Geometry) -> Self {
        Self {
            geometry,
            completed: 0,
            pending: None,
        }
    }
    pub(in crate::resident_leases) fn next(&self) -> Result<Option<Region>> {
        if self.pending.is_some() {
            return Err(Error::State("BF16 source window pending".into()));
        }
        self.geometry.region(self.completed)
    }
    pub(in crate::resident_leases) fn begin(&mut self, region: Region) -> Result<()> {
        if self.next()? != Some(region) {
            return Err(Error::State("BF16 exact next upload region".into()));
        }
        self.pending = Some(region);
        Ok(())
    }
    // Only the original uploader calls this after its exact-view physical barrier.
    pub(in crate::resident_leases) fn checkpoint(&mut self, region: Region) -> Result<()> {
        if self.pending != Some(region) {
            return Err(Error::State("BF16 upload checkpoint differs".into()));
        }
        self.completed = region.end;
        self.pending = None;
        Ok(())
    }
    pub(in crate::resident_leases) fn require_complete(&self) -> Result<()> {
        if self.pending.is_some() || self.completed != self.geometry.rows {
            return Err(Error::State("BF16 authenticated upload incomplete".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_tied_widths_require_every_actual_checkpoint() {
        for (columns, windows) in [(1024, 485), (2048, 970)] {
            let g = Geometry::new(248320, columns).unwrap();
            let mut c = Coverage::new(g);
            let mut bytes = 0;
            for index in 0..windows {
                let r = c.next().unwrap().unwrap();
                assert_eq!(r.offset, bytes);
                assert_eq!(r.bytes, WINDOW);
                for wrong in [
                    Region {
                        first: r.first + 1,
                        ..r
                    },
                    Region {
                        end: r.end - 1,
                        ..r
                    },
                    Region {
                        offset: usize::MAX,
                        ..r
                    },
                    Region {
                        bytes: r.bytes - 1,
                        ..r
                    },
                ] {
                    assert!(c.begin(wrong).is_err());
                    assert_eq!(c.next().unwrap(), Some(r));
                }
                assert!(c.require_complete().is_err());
                assert!(c.checkpoint(r).is_err());
                c.begin(r).unwrap();
                assert!(c.begin(r).is_err());
                assert!(c.require_complete().is_err());
                c.checkpoint(r).unwrap();
                assert!(c.checkpoint(r).is_err());
                bytes += r.bytes;
                if index + 1 == windows {
                    assert_eq!(r.end, 248320);
                }
            }
            assert_eq!(bytes, 248320 * columns * 2);
            assert!(c.next().unwrap().is_none());
            c.require_complete().unwrap();
            c.completed = usize::MAX;
            assert!(c.next().is_err());
            assert!(c.require_complete().is_err());
        }
    }
    #[test]
    fn exact_source_geometry_and_short_tail() {
        let g = Geometry::new(685, 1536).unwrap();
        assert_eq!(g.bytes(), 2_104_320);
        let mut c = Coverage::new(g);
        let mut lengths = Vec::new();
        while let Some(r) = c.next().unwrap() {
            assert!(c.checkpoint(r).is_err());
            c.begin(r).unwrap();
            assert!(c.next().is_err());
            assert!(c.require_complete().is_err());
            assert!(
                c.checkpoint(Region {
                    offset: r.offset + 1,
                    ..r
                })
                .is_err()
            );
            c.checkpoint(r).unwrap();
            assert!(c.checkpoint(r).is_err());
            lengths.push(r.bytes);
        }
        c.require_complete().unwrap();
        assert_eq!(lengths, [1_047_552, 1_047_552, 9_216]);
        assert_eq!(g.output_bytes(&[684, 0, 1, 684]).unwrap(), 12_288);
        assert!(g.output_bytes(&[685]).is_err());
        assert!(g.output_bytes(&[]).is_err());
        assert!(g.output_bytes(&vec![0; WINDOW / 3072 + 1]).is_err());
        assert!(g.portable_limits(g.bytes() as u64 - 1).is_err());
        g.portable_limits(g.bytes() as u64).unwrap();
    }
    #[test]
    fn overflow_alignment_and_monolithic_binding_refuse() {
        for (r, c) in [
            (0, 1024),
            (1, 0),
            (1, 1535),
            (1, 7),
            (1, usize::MAX),
            (usize::MAX, 1024),
        ] {
            assert!(Geometry::new(r, c).is_err());
        }
        let g = Geometry::new(1 << 30, 8).unwrap();
        assert!(g.portable_limits(u64::MAX).is_err());
    }
    #[cfg(feature = "vllm-cuda-qualification")]
    #[test]
    fn cuda_geometry_is_unchanged() {
        for (r, c) in [(248320, 1024), (248320, 2048), (262144, 1536)] {
            let g = Geometry::new(r, c).unwrap();
            let native = g.cuda().unwrap();
            assert_eq!(
                (native.rows(), native.row_bytes(), native.bytes()),
                (r, c * 2, g.bytes())
            );
        }
    }
}
