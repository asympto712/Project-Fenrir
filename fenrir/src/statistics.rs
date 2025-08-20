use crossbeam::channel::{unbounded, Receiver, RecvError, Sender, TryRecvError};
use color_eyre::eyre::{Context, ErrReport};
use std::sync::Arc;

/// (sample size, mean, unbiased variance)
pub type FinalStat = (usize, f64, f64);
/// Almost like FinalStat but the count is in f64 to avoid casting
pub type RunningStat = (f64, f64, f64);

/// Most basic class for statistics recording.
#[derive(Debug, Clone)]
pub struct StatObserver {
    count: f64,
    mean: f64,
    m2: f64,
}

impl StatObserver {

    pub fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0
        }
    }

    /// update the internal statistics
    pub fn update(&mut self, sample: f64) {
        self.count += 1.0;
        let delta = sample - self.mean;
        self.mean += delta / self.count;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;
    }

    /// report the internal statistics
    pub fn report(&self) -> Result<FinalStat, String>{
        if self.count < 2.0 {
            return Err("sample count must be larger than 1".to_string());
        }
        Ok((self.count as usize, self.mean, self.m2 / (self.count - 1.0)))
    }
}

/// Almost like StatObserver, but it has the ability to send its internal statistics to StatCollector
pub struct StatReporter {
    count: f64,
    mean: f64,
    m2: f64,
    sender: Arc<Sender<RunningStat>>
}

impl StatReporter {

    pub fn new(sender: Arc<Sender<RunningStat>>) -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
            sender,
        }
    }

    pub fn update(&mut self, sample: f64) {
        self.count += 1.0;
        let delta = sample - self.mean;
        self.mean += delta / self.count;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sends its statistics to the whoever is on the receiving end
    /// After sending it resets its internal state
    pub fn report(&mut self) -> Result<(),ErrReport>{
        if self.count < 2.0 {
            return Err(ErrReport::msg("stats send failed: sample count must be larger than 1"));
        }
        self.sender.send((self.count, self.mean, self.m2)).wrap_err("stats send failed");
        (self.count, self.mean, self.m2) = (0.0, 0.0, 0.0);
        Ok(())
    }
}

/// Almost like StatObserver, but it holds the receiver of RunningStat and can receive reports from StatReporters
pub struct StatCollector {
    count: f64,
    mean: f64,
    m2: f64,
    receiver: Receiver<RunningStat>
}

impl StatCollector {
    pub fn new(receiver: Receiver<RunningStat>) -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
            receiver
        }
    }

    /// Update the internal stats based on an external stats.
    pub fn update(&mut self, rs: RunningStat) {
        let (count, mean, m2) = rs;
        let new_count= self.count + count;
        let delta = mean - self.mean;
        let count_ratio = count / new_count;

        // update mean
        // following https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm, use different formula
        // for when self.count ~= count
        self.mean = if count_ratio > 0.45 && count_ratio < 0.55 {
            (self.mean * self.count + mean * count) / new_count
        } else {
            self.mean + delta * count_ratio
        };

        self.m2 += m2 + delta * delta * self.count * count_ratio;
        self.count = new_count;
    }

    pub fn recv(&mut self) -> Result<(), RecvError>{
        let (rec_count, rec_mean, rec_m2) = self.receiver.recv()?;
        self.update((rec_count, rec_mean, rec_m2));
        Ok(())
    }

    /// receive a stats without blocking and update its internal state
    pub fn try_recv(&mut self) -> Result<(), TryRecvError> {
        let (rec_count, rec_mean, rec_m2) = self.receiver.try_recv()?;
        self.update((rec_count, rec_mean, rec_m2));
        Ok(())
    }

    /// yield the final result
    pub fn report(&self) -> Result<FinalStat, String>{
        if self.count < 2.0 {
            return Err("sample count must be larger than 1".to_string());
        }
        Ok((self.count as usize, self.mean, self.m2 / (self.count - 1.0)))
    }
}