use crate::device::{
    DeviceId, DeviceService,
    handle::{CallError, DeviceHandleSpec, ServerUtilitiesHandle, ServiceCreationError},
};
use alloc::boxed::Box;
use core::any::{Any, TypeId};
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{Arc, Mutex, RwLock};
use hashbrown::HashMap;

/// A handle to a specific device context (no-std version).
pub struct MutexDeviceHandle {
    state: MutexDeviceState,
    device_id: DeviceId,
}

#[derive(Clone)]
struct MutexDeviceState {
    service: Arc<Mutex<Box<dyn Any + Send>>>,
    device_id: DeviceId,
    utilities: ServerUtilitiesHandle,
}

/// The global storage for all device services.
/// In no-std, we use a global registry protected by a Mutex.
static DEVICE_REGISTRY: spin::Mutex<Option<HashMap<DeviceId, DeviceRegistry>>> =
    spin::Mutex::new(None);

/// Maps `TypeId` to the actual Service instance.
type DeviceRegistry = HashMap<TypeId, MutexDeviceState>;

impl DeviceHandleSpec for MutexDeviceHandle {
    const BLOCKING: bool = true;

    fn new<S: DeviceService>(device_id: DeviceId) -> Self {
        let mut guard = DEVICE_REGISTRY.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };
        let device_map: &mut HashMap<_, _> = match guard.as_mut() {
            Some(val) => val.entry(device_id).or_insert_with(HashMap::new),
            None => unreachable!(),
        };

        let type_id = TypeId::of::<S>();

        let state = device_map
            .entry(type_id)
            .or_insert_with(|| {
                let state = S::init(device_id);
                let utilities = state.utilities();
                MutexDeviceState {
                    service: Arc::new(Mutex::new(Box::new(state))),
                    device_id,
                    utilities,
                }
            })
            .clone();

        Self { state, device_id }
    }

    fn get_existing<S: DeviceService>(device_id: DeviceId) -> Option<Self> {
        let state = DEVICE_REGISTRY
            .lock()
            .as_ref()?
            .get(&device_id)?
            .get(&TypeId::of::<S>())?
            .clone();
        Some(Self { state, device_id })
    }

    fn get_or_try_insert_with<S: DeviceService>(
        device_id: DeviceId,
        create: impl FnOnce() -> Result<S, ServiceCreationError>,
    ) -> Result<Self, ServiceCreationError> {
        Self::insert_with(device_id, false, create)
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.state.utilities.clone()
    }

    fn flush_queue(&self) {}

    fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let mut guard = self.state.service.lock();

        Ok(task(&mut **guard))
    }

    fn submit<T: FnOnce(&mut dyn Any) + Send + 'static>(&self, task: T) {
        let mut guard = self.state.service.lock();

        task(&mut **guard);
    }

    fn insert<S: DeviceService>(
        device_id: DeviceId,
        service: S,
    ) -> Result<Self, ServiceCreationError> {
        Self::insert_with(device_id, true, || Ok(service))
    }

    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let lock = self.device_lock();
        let guard = lock.lock();
        let result = Ok(task());
        core::mem::drop(guard);
        result
    }
}

impl MutexDeviceHandle {
    fn insert_with<S: DeviceService>(
        device_id: DeviceId,
        reject_existing: bool,
        create: impl FnOnce() -> Result<S, ServiceCreationError>,
    ) -> Result<Self, ServiceCreationError> {
        let mut guard = DEVICE_REGISTRY.lock();
        let map = guard
            .get_or_insert_with(HashMap::new)
            .entry(device_id)
            .or_default();
        let id = TypeId::of::<S>();
        if let Some(state) = map.get(&id) {
            if reject_existing {
                return Err(ServiceCreationError::new("Service already created".into()));
            }
            return Ok(Self {
                state: state.clone(),
                device_id,
            });
        }
        let service = create()?;
        let state = MutexDeviceState {
            utilities: service.utilities(),
            service: Arc::new(Mutex::new(Box::new(service))),
            device_id,
        };
        map.insert(id, state.clone());
        Ok(Self { state, device_id })
    }
}

static DEVICE_LOCK: spin::Mutex<Option<HashMap<DeviceId, Arc<DeviceLock>>>> =
    spin::Mutex::new(None);

struct DeviceLock {
    lock: RwLock<Option<StreamId>>,
    main: spin::Mutex<()>,
}

enum Guard<'a> {
    Reentrant,
    Main(spin::MutexGuard<'a, ()>, &'a DeviceLock),
}

impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        match self {
            Guard::Reentrant => {}
            Guard::Main(_mutex_guard, thread_mutex) => {
                let mut state = thread_mutex.lock.write();
                *state = None;
            }
        }
    }
}

impl DeviceLock {
    pub fn lock(&self) -> Guard<'_> {
        // TODO: Use thread id when we can.
        let stream_id = StreamId::current();

        loop {
            let mut state = self.lock.write();

            let is_ok = match state.as_ref() {
                Some(value) => *value == stream_id,
                None => {
                    *state = Some(stream_id);
                    let guard = self.main.lock();
                    return Guard::Main(guard, self);
                }
            };

            match is_ok {
                true => {
                    core::mem::drop(state);
                    return Guard::Reentrant;
                }
                false => {
                    // spin.
                }
            };
        }
    }
}

impl MutexDeviceHandle {
    fn device_lock(&self) -> Arc<DeviceLock> {
        let mut guard = DEVICE_LOCK.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };

        let device_map = match guard.as_mut() {
            Some(val) => val.entry(self.device_id),
            None => unreachable!(),
        };

        device_map
            .or_insert_with(|| {
                Arc::new(DeviceLock {
                    lock: RwLock::new(None),
                    main: spin::Mutex::new(()),
                })
            })
            .clone()
    }
}
impl Clone for MutexDeviceHandle {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            device_id: self.device_id,
        }
    }
}
