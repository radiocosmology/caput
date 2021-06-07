// 2**31 + 2**30 will be used to check for overflow
const uint32_t HIGH_BITS = 3221225472;

// 2**63 + 2**62 will be used to check for overflow
const uint64_t HIGH_BITS_DOUBLE = 13835058055282163712UL;

// The length of the part in a float that represents the exponent
const int32_t LEN_EXPONENT_FLOAT = 8;

// The length of the part in a double that represents the exponent
const int64_t LEN_EXPONENT_DOUBLE = 11;

// Starting bit (offset) of the part in a float that represents the exponent
const int32_t POS_EXPONENT_FLOAT = 23;

// Starting bit (offset) of the part in a double that represents the exponent
const int64_t POS_EXPONENT_DOUBLE = 52;

// A mask to apply on the exponent representation of a float, to get rid of the sign part
const int32_t MASK_EXPONENT_W_O_SIGN_FLOAT = 255;

// A mask to apply on the exponent representation of a double, to get rid of the sign part
const int64_t MASK_EXPONENT_W_O_SIGN_DOUBLE = 2047;

// A mask to apply on a float to get only the mantissa (2**23 - 1)
const int32_t MASK_MANTISSA_FLOAT = 8388607;

// A mask to apply on a double to get only the mantissa (2**52 - 1)
const int64_t MASK_MANTISSA_DOUBLE = 4503599627370495L;

// Implicit 24th bit of the mantissa in a float (2**23)
const int32_t IMPLICIT_BIT_FLOAT = 8388608;

// Implicit 53rt bit of the mantissa in a double (2**52)
const int64_t IMPLICIT_BIT_DOUBLE = 4503599627370496L;

// The maximum error we can have for the mantissa in a float (less than 2**30)
const int32_t ERR_MAX_FLOAT = 1073741824;

// The maximum error we can have for the mantissa in a double (less than 2**30)
const int64_t ERR_MAX_DOUBLE = 4611686018427387903L;

/**
 *  @brief Truncate the precision of *val* by rounding to a multiple of a power of
 *        two, keeping error less than or equal to *err*.
 *
 *  @warning Undefined results for err < 0 and err > 2**30.
 */
inline int32_t bit_truncate(int32_t val, int32_t err) {
    // *gran* is the granularity. It is the power of 2 that is *larger than* the
    // maximum error *err*.
    int32_t gran = err;
    gran |= gran >> 1;
    gran |= gran >> 2;
    gran |= gran >> 4;
    gran |= gran >> 8;
    gran |= gran >> 16;
    gran += 1;

    // Bitmask selects bits to be rounded.
    int32_t bitmask = gran - 1;

    // Determine if there is a round-up/round-down tie.
    // This operation gets the `gran = 1` case correct (non tie).
    int32_t tie = ((val & bitmask) << 1) == gran;

    // The acctual rounding.
    int32_t val_t = (val - (gran >> 1)) | bitmask;
    val_t += 1;
    // There is a bit of extra bit twiddling for the err == 0.
    val_t -= (err == 0);

    // Break any tie by rounding to even.
    val_t -= val_t & (tie * gran);

    return val_t;
}


/**
 *  @brief Truncate the precision of *val* by rounding to a multiple of a power of
 *        two, keeping error less than or equal to *err*.
 *
 *  @warning Undefined results for err < 0 and err > 2**62.
 */
inline int64_t bit_truncate_64(int64_t val, int64_t err) {
    // *gran* is the granularity. It is the power of 2 that is *larger than* the
    // maximum error *err*.
    int64_t gran = err;
    gran |= gran >> 1;
    gran |= gran >> 2;
    gran |= gran >> 4;
    gran |= gran >> 8;
    gran |= gran >> 16;
    gran |= gran >> 32;
    gran += 1;

    // Bitmask selects bits to be rounded.
    int64_t bitmask = gran - 1;

    // Determine if there is a round-up/round-down tie.
    // This operation gets the `gran = 1` case correct (non tie).
    int64_t tie = ((val & bitmask) << 1) == gran;

    // The acctual rounding.
    int64_t val_t = (val - (gran >> 1)) | bitmask;
    val_t += 1;
    // There is a bit of extra bit twiddling for the err == 0.
    val_t -= (err == 0);

    // Break any tie by rounding to even.
    val_t -= val_t & (tie * gran);

    return val_t;
}


/**
 *  @brief Count the number of leading zeros in a binary number.
 *         Taken from https://stackoverflow.com/a/23857066
 */
inline int32_t count_zeros(int32_t x) {
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return __builtin_popcount(~x);
}


/**
 *  @brief Count the number of leading zeros in a binary number.
 *         Taken from https://stackoverflow.com/a/23857066
 */
inline int64_t count_zeros_64(int64_t x) {
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    x = x | (x >> 32);
    return __builtin_popcountl(~x);
}


/**
 * @brief Fast power of two float.
 *
 * Result is undefined for e < -126.
 *
 * @param   e   Exponent
 *
 * @returns The result of 2^e
 */
inline float fast_pow(int8_t e) {
    float* out_f;
    // Construct float bitwise
    uint32_t out_i = ((uint32_t)(127 + e) << 23);
    // Cast into float
    out_f = (float*)&out_i;
    return *out_f;
}


/**
 * @brief Fast power of two double.
 *
 * Result is undefined for e < -1022 and e > 1023.
 *
 * @param   e   Exponent
 *
 * @returns The result of 2^e
 */
inline double fast_pow_double(int16_t e) {
    double* out_f;
    // Construct float bitwise
    uint64_t out_i = ((uint64_t)(1023 + e) << 52);
    // Cast into float
    out_f = (double*)&out_i;
    return *out_f;
}


/**
 *  @brief Truncate precision of a floating point number by applying the algorithm of
 *         `bit_truncate` to the mantissa.
 *
 *  Note that NaN and inf are not explicitly checked for. According to the IEEE spec, it is
 *  impossible for the truncation to turn an inf into a NaN. However, if the truncation
 *  happens to remove all of the non-zero bits in the mantissa, a NaN can become inf.
 *
 */
inline float _bit_truncate_float(float val, float err) {
    // cast float memory into an int
    int32_t* cast_val_ptr = (int32_t*)&val;
    // extract the exponent and sign
    int32_t val_pre = cast_val_ptr[0] >> POS_EXPONENT_FLOAT;
    // strip sign
    int32_t val_pow = val_pre & MASK_EXPONENT_W_O_SIGN_FLOAT;
    int32_t val_s = val_pre >> LEN_EXPONENT_FLOAT;
    // extract mantissa. mask is 2**23 - 1. Add back the implicit 24th bit
    int32_t val_man = (cast_val_ptr[0] & MASK_MANTISSA_FLOAT) + IMPLICIT_BIT_FLOAT;
    // scale the error to the integer representation of the mantissa
    // scale by 2**(23 + 127 - pow)
    int32_t int_err = (int32_t)(err * fast_pow(150 - val_pow));
    // make sure hasn't overflowed. if set to 2**30-1, will surely round to 0.
    // must keep err < 2**30 for bit_truncate to work
    int_err = (int_err & HIGH_BITS) ? ERR_MAX_FLOAT : int_err;

    // truncate
    int32_t tr_man = bit_truncate(val_man, int_err);

    // count leading zeros
    int32_t z_count = count_zeros(tr_man);
    // adjust power after truncation to account for loss of implicit bit
    val_pow -= z_count - 8;
    // shift mantissa by same amount, remove implicit bit
    tr_man = (tr_man << (z_count - 8)) & MASK_MANTISSA_FLOAT;
    // round to zero case
    val_pow = ((z_count != 32) ? val_pow : 0);
    // restore sign and exponent
    int32_t tr_val = tr_man | ((val_pow | (val_s << 8)) << 23);
    // cast back to float
    float* tr_val_ptr = (float*)&tr_val;

    return tr_val_ptr[0];
}


/**
 *  @brief Truncate precision of a double floating point number by applying the algorithm of
 *         `bit_truncate` to the mantissa.
 *
 *  Note that NaN and inf are not explicitly checked for. According to the IEEE spec, it is
 *  impossible for the truncation to turn an inf into a NaN. However, if the truncation
 *  happens to remove all of the non-zero bits in the mantissa, a NaN can become inf.
 *
 */
inline double _bit_truncate_double(double val, double err) {
    // Step 1: Extract the sign, exponent and mantissa:
    // ------------------------------------------------
    // cast float memory into an int
    int64_t* cast_val_ptr = (int64_t*)&val;
    // extract the exponent and sign
    int64_t val_pre = cast_val_ptr[0] >> POS_EXPONENT_DOUBLE;
    // strip sign
    int64_t val_pow = val_pre & MASK_EXPONENT_W_O_SIGN_DOUBLE;
    int64_t val_s = val_pre >> LEN_EXPONENT_DOUBLE;
    // extract mantissa. mask is 2**52 - 1. Add back the implicit 53rd bit
    int64_t val_man = (cast_val_ptr[0] & MASK_MANTISSA_DOUBLE) + IMPLICIT_BIT_DOUBLE;

    // Step 2: Scale the error to the integer representation of the mantissa:
    // ----------------------------------------------------------------------
    // scale by 2**(52 + 1023 - pow)
    int64_t int_err = (int64_t)(err * fast_pow_double(1075 - val_pow));
    // make sure hasn't overflowed. if set to 2**62-1, will surely round to 0.
    // must keep err < 2**62 for bit_truncate_double to work
    int_err = (int_err & HIGH_BITS_DOUBLE) ? ERR_MAX_DOUBLE : int_err;

    // Step 3: Truncate the mantissa:
    // ------------------------------
    int64_t tr_man = bit_truncate_64(val_man, int_err);

    // Step 4: Put it back together:
    // -----------------------------
    // count leading zeros
    int64_t z_count = count_zeros_64(tr_man);
    // adjust power after truncation to account for loss of implicit bit
    val_pow -= z_count - 11;
    // shift mantissa by same amount, remove implicit bit
    tr_man = (tr_man << (z_count - 11)) & MASK_MANTISSA_DOUBLE;
    // round to zero case
    val_pow = ((z_count != 64) ? val_pow : 0);
    // restore sign and exponent
    int64_t tr_val = tr_man | ((val_pow | (val_s << 11)) << 52);
    // cast back to double
    double* tr_val_ptr = (double*)&tr_val;

    return tr_val_ptr[0];
}
