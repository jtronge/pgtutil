#ifndef PGTUTIL_COMM_H
#define PGTUTIL_COMM_H

namespace pgtutil::comm {

// MPI window RMA communication abstraction
struct Window {
    MPI_Win win;
    uint64_t* ptr;

    Window(MPI_Comm comm, std::span<const uint64_t> data)
    {
        // Allocate the window
        MPI_Win_allocate(data.size() * sizeof(uint64_t), sizeof(uint64_t),
                         MPI_INFO_NULL, comm, &ptr, &win);
        // Copy over the data
        MPI_Win_fence(0, win);
        for (uint64_t i = 0; i < data.size(); ++i) {
            ptr[i] = data[i];
        }
        MPI_Win_fence(0, win);
    }

    ~Window()
    {
        MPI_Win_free(&win);
    }

    // Open and close fence calls for the MPI window
    void fence()
    {
        MPI_Win_fence(0, win);
    }

    // Put data to a remote window
    void put(std::span<const uint64_t> buf, int dest, uint64_t off)
    {
        MPI_Put(buf.data(), buf.size(), MPI_UINT64_T,
                dest, off, buf.size(), MPI_UINT64_T, win);
    }

    // Get data from a remote window
    void get(std::span<uint64_t> buf, int src, uint64_t off)
    {
        MPI_Get(buf.data(), buf.size(), MPI_UINT64_T,
                src, off, buf.size(), MPI_UINT64_T, win);
    }
};

}

#endif
