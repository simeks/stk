namespace stk
{
    template<typename T>
    VolumeHelper<T> normalize(const VolumeHelper<T>& src,
                              T min,
                              T max,
                              VolumeHelper<T>* out)
    {
        T src_min, src_max;
        find_min_max(src, src_min, src_max);

        dim3 dims = src.size();

        VolumeHelper<T> dest;
        if (!out) {
            dest.allocate(src.size());
            out = &dest;
        }
        ASSERT(out->size() == dims);

        out->copy_meta_from(src);

        double range = double(max - min);
        double src_range = double(src_max - src_min);

        #pragma omp parallel for
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    (*out)(x,y,z) = T(range * (src(x, y, z) - src_min) / src_range + min);
                }
            }
        }
        return *out;
    }
}
