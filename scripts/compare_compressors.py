import sys
import zarr
from numcodecs import Blosc, BZ2, GZip, register_codec
from time import perf_counter


compressors = {
    'Blosc': lambda clevel: Blosc(cname='blosclz', clevel=clevel, shuffle=Blosc.SHUFFLE),
    'Zstd': lambda clevel: Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.SHUFFLE),
    'LZ4': lambda clevel: Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.SHUFFLE),
    'LZ4hc': lambda clevel: Blosc(cname='lz4hc', clevel=clevel, shuffle=Blosc.SHUFFLE),
    'Zlib': lambda clevel: Blosc(cname='zlib', clevel=clevel, shuffle=Blosc.SHUFFLE),
    'Snappy': lambda clevel: Blosc(cname='snappy', clevel=clevel, shuffle=Blosc.SHUFFLE),
    'BZ2': lambda clevel: BZ2(level=clevel),
    'GZip': lambda clevel: GZip(level=clevel)
}


def test_compressor(z, comp_name, comp_level=5, chunk_size=1024):
    H = z['compressed'].attrs['compression_metadata']['height']
    W = z['compressed'].attrs['compression_metadata']['width']
    channels = z['compressed'].attrs['compression_metadata']['compressed_channels']

    compressor = compressors[comp_name](comp_level)
    e_time = perf_counter()
    tmp_zarr = zarr.array(z['compressed/0/0'][:], chunks=(1, channels, 1, chunk_size, chunk_size), dtype='u1', compressor=compressor)
    e_time = perf_counter() - e_time
    print('Array shape {} ({} bytes) compressed into {} bytes using {} with compression level {}. BPP={}. Execution time={}'.format(z['compressed/0/0'].shape, z['compressed/0/0'].nbytes, tmp_zarr.nbytes_stored, comp_name, comp_level, tmp_zarr.nbytes_stored/(H * W), e_time))

if __name__ == "__main__":
    print('Test compressors for zarr files')

    z_fn = sys.argv[1]
    z_chunk = int(sys.argv[2])
    z_clevel = int(sys.argv[3])

    z = zarr.open(z_fn, mode='r')

    for comp_name in compressors.keys():
        test_compressor(z, comp_name, z_clevel, z_chunk)
