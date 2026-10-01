#ifndef PGTUTIL_COMM_H
#define PGTUTIL_COMM_H

namespace pgtutil::comm {

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<uint64_t>()
{
    return MPI_UINT64_T;
}

template<>
MPI_Datatype get_mpi_type<double>()
{
    return MPI_DOUBLE;
}

// MPI window RMA communication abstraction

// MPI window RMA communication abstraction
template<typename T>
struct Window {
    MPI_Win win;
    T* ptr;

    Window(MPI_Comm comm, std::span<const T> data)
    {
        // Allocate the window
        MPI_Win_allocate(data.size() * sizeof(T), sizeof(T),
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
    void put(std::span<const T> buf, int dest, uint64_t off)
    {
        MPI_Put(buf.data(), buf.size(), get_mpi_type<T>(),
                dest, off, buf.size(), get_mpi_type<T>(), win);
    }

    // Get data from a remote window
    void get(std::span<T> buf, int src, uint64_t off)
    {
        MPI_Get(buf.data(), buf.size(), get_mpi_type<T>(),
                src, off, buf.size(), get_mpi_type<T>(), win);
    }
};

}

#endif
