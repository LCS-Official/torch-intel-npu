/* Four metadata words, row-major FP32 values; optional raw INT64 candidates
 * in the second input. Output contains row values followed by INT64 indices.
 * Compare ordered bits so subnormals and the first NaN/tie survive SHAVE FTZ.
 */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned size = 0;
    for (unsigned t = 0; t < 3; ++t) {
        if (!m[t].isStatic || m[t].rank != 4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype != 1) return;
        unsigned count = 1;
        const unsigned *dims = (const unsigned*)m[t].dims;
        for (unsigned a = 0; a < 4; ++a) {
            if (!dims[a] || dims[a] > 65536u/count) return;
            count *= dims[a];
        }
        if (t && size != count) return;
        size = count;
    }
    if (size < 5) return;
    const float *header = (const float*)m[0].lo;
    if (!(header[0] >= 1 && header[0] <= 21845 && header[1] >= 1 && header[1] <= 65532 &&
          header[2] >= 0 && header[2] <= 1048576 && (header[3] == 0 || header[3] == 1))) return;
    unsigned rows = (unsigned)header[0], width = (unsigned)header[1], offset = (unsigned)header[2];
    if ((float)rows != header[0] || (float)width != header[1] || (float)offset != header[2] ||
        rows > (size-4)/width || rows > size/3 || offset+width > 1048576 ||
        (header[3] && rows*width > size/2)) return;
    const unsigned *values = (const unsigned*)(header+4), *ids = (const unsigned*)m[1].lo;
    unsigned *out = (unsigned*)m[2].lo;
    for (unsigned i = 0; i < size; ++i) out[i] = 0;
    for (unsigned row = 0; row < rows; ++row) {
        unsigned best = values[row*width], winner = 0;
        for (unsigned j = 1; j < width; ++j) {
            unsigned v = values[row*width+j], a = best & 0x7fffffffu, b = v & 0x7fffffffu;
            unsigned ka = best & 0x80000000u ? ~best : best ^ 0x80000000u;
            unsigned kb = v & 0x80000000u ? ~v : v ^ 0x80000000u;
            if (a <= 0x7f800000u && (b > 0x7f800000u ||
                ((a || b) && (ROW_MIN ? kb < ka : kb > ka)))) { best = v; winner = j; }
        }
        out[row] = best;
        unsigned index = row*width+winner;
        out[rows+2*row] = header[3] ? ids[2*index] : offset+winner;
        out[rows+2*row+1] = header[3] ? ids[2*index+1] : 0;
    }
}
