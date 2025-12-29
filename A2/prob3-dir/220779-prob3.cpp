#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <vector>
#include <climits>

using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define N (1e7)
int NUM_THREADS;
constexpr int MAX_THREADS = 64;

// Cache line size for preventing false sharing
constexpr size_t CACHE_LINE_SIZE = 64;

// Padding structure to prevent false sharing
struct alignas(CACHE_LINE_SIZE) PaddedInt {
    volatile int value;
    char padding[CACHE_LINE_SIZE - sizeof(int)];
    PaddedInt() : value(0) {}
    PaddedInt(int v) : value(v) {}
};

struct alignas(CACHE_LINE_SIZE) PaddedUint64 {
    volatile uint64_t value;
    char padding[CACHE_LINE_SIZE - sizeof(uint64_t)];
    PaddedUint64() : value(0) {}
    PaddedUint64(uint64_t v) : value(v) {}
};

struct alignas(CACHE_LINE_SIZE) PaddedUint32 {
    volatile int value;
    char padding[CACHE_LINE_SIZE - sizeof(int)];
    PaddedUint32() : value(0) {}
    PaddedUint32(int v) : value(v) {}
};

struct alignas(CACHE_LINE_SIZE) PaddedBool {
    volatile bool value;
    char padding[CACHE_LINE_SIZE - sizeof(bool)];
    PaddedBool() : value(false) {}
    PaddedBool(bool v) : value(v) {}
};

// Shared variables - separated to prevent false sharing
alignas(CACHE_LINE_SIZE) uint64_t var1;
alignas(CACHE_LINE_SIZE) uint64_t var2;

// ----------- GLOBAL CAS HELPER FUNCTIONS -----------
// Custom helper function using cmpxchg instruction for atomic compare-and-swap
inline bool atomic_cmpxchg_int(volatile int* addr, int expected, int new_val) {
    int result;
    __asm__ __volatile__(
        "lock cmpxchgl %2, %1"    // Compare EAX with memory, if equal set memory to new_val
        : "=a"(result), "+m"(*addr)  // output: result in EAX, memory operand
        : "r"(new_val), "a"(expected)  // input: new_val in register, expected in EAX
        : "memory"                
    );
    return result == expected;  // Return true if swap succeeded
}

// CAS-based fetch-and-add operation for int
inline int cas_fetch_add(volatile int* addr, int increment) {
    int old_val, new_val;
    do {
        old_val = *addr;
        new_val = old_val + increment;
    } while (!atomic_cmpxchg_int(addr, old_val, new_val));
    __asm__ __volatile__("mfence" ::: "memory");  // Full memory barrier
    return old_val;  // Return the old value (like fetch_add)
}


// Abstract base class
class LockBase {
public:
  // Pure virtual function
  virtual void acquire(uint16_t tid) = 0;
  virtual void release(uint16_t tid) = 0;
};

typedef struct alignas(CACHE_LINE_SIZE) thr_args {
  uint16_t m_id;
  LockBase* m_lock;
  char padding[CACHE_LINE_SIZE - sizeof(uint16_t) - sizeof(LockBase*)];
} ThreadArgs;

/** Use pthread mutex to implement lock routines */
class PthreadMutex : public LockBase {
public:
  void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
  void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }

private:
  pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

// ----------- FILTER LOCK -----------
class FilterLock : public LockBase {
public:
    FilterLock() {
        int n = NUM_THREADS;
        level = new PaddedInt[n];
        victim = new PaddedInt[n]; 
        for (int i = 0; i < n; ++i) {
            level[i].value = 0;
            victim[i].value = -1;
        }
    }
    ~FilterLock() {
        delete[] level;
        delete[] victim;
    }
    
    void acquire(uint16_t tid) override {
        int n = NUM_THREADS;
        for (int L = 1; L < n; ++L) {
            level[tid].value = L;
            __asm__ __volatile__("mfence" ::: "memory");
            victim[L].value = tid;
            __asm__ __volatile__("mfence" ::: "memory");

            // Wait until no thread has level >= L and I'm still the victim
            for (int k = 0; k < n; ++k) {
                if (k == tid) continue;
                while (level[k].value >= L && victim[L].value == tid) {
                    // spin - use pause instruction to reduce CPU contention
                    __asm__ __volatile__("pause" ::: "memory");
                }
            }
        }
    }
    
    void release(uint16_t tid) override {
        level[tid].value = 0;
        __asm__ __volatile__("mfence" ::: "memory");        
    }
    
private:
    PaddedInt* level;
    PaddedInt* victim;
};

// ----------- BAKERY LOCK -----------
class BakeryLock : public LockBase {
public:
    BakeryLock() {
        int n = NUM_THREADS;
        choosing = new PaddedInt[n];
        number = new PaddedUint64[n];
        for (int i = 0; i < n; ++i) {
            choosing[i].value = 0;  
            number[i].value = 0;
        }
    }
    ~BakeryLock() {
        delete[] choosing;
        delete[] number;
    }
    
    void acquire(uint16_t tid) override {
        int n = NUM_THREADS;
        
        choosing[tid].value = 1;
        __asm__ __volatile__("mfence" ::: "memory");
        
        // Find maximum number
        uint64_t maxnum = 0;
        for (int i = 0; i < n; ++i) {
            uint64_t nval = number[i].value;
            if (nval > maxnum) maxnum = nval;
        }
        number[tid].value = maxnum + 1;
        __asm__ __volatile__("mfence" ::: "memory");
        
        choosing[tid].value = 0;
        __asm__ __volatile__("mfence" ::: "memory");

        // Wait for all threads with smaller (priority, id) pairs
        for (int j = 0; j < n; ++j) {
            if (j == tid) continue;
            
            // Wait while thread j is choosing its number
            while (choosing[j].value != 0) {
                __asm__ __volatile__("pause" ::: "memory");
            }
            
            // Wait while thread j has a ticket and (j,number[j]) < (tid,number[tid])
            while (true) {
                uint64_t nj = number[j].value;
                uint64_t ntid = number[tid].value;

                if (nj == 0) break; 
                
                if (nj < ntid || (nj == ntid && j < tid)) {
                    __asm__ __volatile__("pause" ::: "memory");
                } else {
                    break;
                }
            }
        }
    }
    
    void release(uint16_t tid) override {
        number[tid].value = 0;
        __asm__ __volatile__("mfence" ::: "memory");
    }
    
private:
    PaddedInt* choosing;
    PaddedUint64* number;
};

// ----------- SPIN LOCK -----------
class SpinLock : public LockBase {
public:
    SpinLock() : flag(0) {}
    
    void acquire(uint16_t) override {
        // Keep trying to set flag from 0 to 1 using custom cmpxchg
        while (!atomic_cmpxchg_int(&flag, 0, 1)) {
            __asm__ __volatile__("pause" ::: "memory");
        }
        __asm__ __volatile__("mfence" ::: "memory");
    }
    
    void release(uint16_t) override {
        __asm__ __volatile__("mfence" ::: "memory");
        flag = 0;
        __asm__ __volatile__("mfence" ::: "memory");

    }
    
private:
    volatile int flag;
};

// ----------- TICKET LOCK -----------
class TicketLock : public LockBase {
public:
    TicketLock() : next_ticket(0), now_serving(0) {}
    
    void acquire(uint16_t) override {
        // Atomically get my ticket number using CAS-based fetch-and-add
        int my_ticket = cas_fetch_add(&next_ticket.value, 1);
        
        // Wait until it's my turn
        while (now_serving.value != my_ticket) {
            __asm__ __volatile__("pause" ::: "memory");
        }
    }
    
    void release(uint16_t) override {
        // Increment now_serving to allow next thread to proceed
        cas_fetch_add(&now_serving.value, 1);
    }
    
private:
    PaddedUint32 next_ticket;
    PaddedUint32 now_serving;
};

class ArrayQLock : public LockBase {
public:
    ArrayQLock() : tail(0) {
        // Initialize all flags to false except the first one
        for (int i = 0; i < MAX_THREADS; ++i) {
            flags[i].value = false;
        }
        flags[0].value = true; 
        
        // Initialize my_slot array
        for (int i = 0; i < MAX_THREADS; ++i) {
            my_slot[i] = 0;
        }
    }
    
    void acquire(uint16_t tid) override {
        // Atomically get my slot number using CAS-based fetch-and-add
        int slot = cas_fetch_add(&tail.value, 1) % NUM_THREADS;
        my_slot[tid] = slot;
        
        // Wait until my slot becomes available
        while (!flags[slot].value) {
            __asm__ __volatile__("pause" ::: "memory");
        }
    }
    
    void release(uint16_t tid) override {
        int slot = my_slot[tid];
        flags[slot].value = false;
        flags[(slot + 1) % NUM_THREADS].value = true;

    }
    
private:
    PaddedUint32 tail;
    PaddedBool flags[MAX_THREADS];
    int my_slot[MAX_THREADS];
};

/** Estimate the time taken */
std::atomic_uint64_t sync_time = 0;

inline void critical_section() {
  var1++;
  var2--;
}

/** Sync threads at the start to maximize contention */
pthread_barrier_t g_barrier;

void* thrBody(void* arguments) {
  ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
  if (false) {
    cout << "Thread id: " << tmp->m_id << " starting\n";
  }

  // Wait for all other producer threads to launch before proceeding.
  pthread_barrier_wait(&g_barrier);

  HRTimer start = HR::now();
  for (int i = 0; i < N; i++) {
    tmp->m_lock->acquire(tmp->m_id);
    critical_section();
    tmp->m_lock->release(tmp->m_id);
  }
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  // A barrier is not required here
  sync_time.fetch_add(duration);
  pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
  if(argc!=2){
    std::cerr<<"Usage: ./a.out <NUM_THREADS>\n";
    exit(EXIT_FAILURE);
  }
  NUM_THREADS = atoi(argv[1]);
  if (NUM_THREADS <= 0 || NUM_THREADS > 64) {
      cerr << "Invalid number of threads. Allowed range: [0,64] (powers of 2).\n";
      exit(EXIT_FAILURE);
  }
  int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
  if (error != 0) {
    cerr << "Error in barrier init.\n";
    exit(EXIT_FAILURE);
  }

  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t tid[NUM_THREADS];
  ThreadArgs args[NUM_THREADS] = {{0}};

  // Pthread mutex
  LockBase* lock_obj = new PthreadMutex();
  uint16_t i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      cerr << "\nThread cannot be created : " << strerror(error) << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  void* status;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      cerr << "ERROR: return code from pthread_join() is " << error << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }

  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Pthread mutex: Time taken (us): " << sync_time << "\n";

  // Filter lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new FilterLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Filter lock: Time taken (us): " << sync_time << "\n";

  // Bakery lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new BakeryLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Bakery lock: Time taken (us): " << sync_time << "\n";

  // Spin lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new SpinLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Spin lock: Time taken (us): " << sync_time << "\n";

  // Ticket lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new TicketLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Ticket lock: Time taken (us): " << sync_time << "\n";

  // Array Q lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new ArrayQLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Array Q lock: Time taken (us): " << sync_time << "\n";

  pthread_barrier_destroy(&g_barrier);
  pthread_attr_destroy(&attr);

  pthread_exit(NULL);
}
