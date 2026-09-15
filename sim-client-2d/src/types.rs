#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ActivePreset {
    BeachStream,
    BeachWaves,
    BeachWaves2,
    DamBreak,
    LakeAtRest,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FlowVisMode {
    Both,
    Vectors,
    Particles,
    Off,
}

impl FlowVisMode {
    pub fn next(self) -> Self {
        match self {
            Self::Particles => Self::Both,
            Self::Both => Self::Vectors,
            Self::Vectors => Self::Off,
            Self::Off => Self::Particles,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Both => "Flow Lines & Particles",
            Self::Vectors => "Flow Lines (Vectors)",
            Self::Particles => "Particles (Tracers)",
            Self::Off => "Off",
        }
    }
}

pub struct FastRng(pub u32);

impl FastRng {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00FF_FFFF) as f32 / 16777216.0
    }

    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}
