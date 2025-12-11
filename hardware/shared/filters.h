/**
 * AI Boxing Trainer - Signal Filters
 * Low-pass and moving average filters for IMU data
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <cstring>

// =============================================================================
// Moving Average Filter
// =============================================================================

template<typename T, size_t SIZE>
class MovingAverageFilter {
private:
    T buffer[SIZE];
    size_t index;
    size_t count;
    T sum;

public:
    MovingAverageFilter() : index(0), count(0), sum(0) {
        memset(buffer, 0, sizeof(buffer));
    }

    T update(T value) {
        // Subtract oldest value from sum
        sum -= buffer[index];

        // Add new value
        buffer[index] = value;
        sum += value;

        // Update index and count
        index = (index + 1) % SIZE;
        if (count < SIZE) count++;

        // Return average
        return sum / (T)count;
    }

    T get() const {
        return count > 0 ? sum / (T)count : 0;
    }

    void reset() {
        memset(buffer, 0, sizeof(buffer));
        index = 0;
        count = 0;
        sum = 0;
    }
};

// =============================================================================
// Exponential Moving Average (EMA) Filter
// =============================================================================

class EMAFilter {
private:
    float alpha;
    float value;
    bool initialized;

public:
    EMAFilter(float smoothing_factor = 0.2f)
        : alpha(smoothing_factor), value(0), initialized(false) {}

    float update(float input) {
        if (!initialized) {
            value = input;
            initialized = true;
        } else {
            value = alpha * input + (1.0f - alpha) * value;
        }
        return value;
    }

    float get() const { return value; }

    void reset() {
        value = 0;
        initialized = false;
    }

    void setAlpha(float a) { alpha = a; }
};

// =============================================================================
// Low-Pass Filter (Butterworth-style)
// =============================================================================

class LowPassFilter {
private:
    float cutoff_freq;
    float sample_rate;
    float prev_input;
    float prev_output;
    float a0, a1, b1;
    bool initialized;

public:
    LowPassFilter(float cutoff = 10.0f, float sample_rate = 100.0f)
        : cutoff_freq(cutoff), sample_rate(sample_rate),
          prev_input(0), prev_output(0), initialized(false) {
        // Calculate filter coefficients
        float omega = 2.0f * 3.14159265f * cutoff_freq / sample_rate;
        float alpha = omega / (omega + 1.0f);
        a0 = alpha;
        a1 = alpha;
        b1 = 1.0f - alpha;
    }

    float update(float input) {
        if (!initialized) {
            prev_input = input;
            prev_output = input;
            initialized = true;
            return input;
        }

        float output = a0 * input + a1 * prev_input + b1 * prev_output;
        prev_input = input;
        prev_output = output;
        return output;
    }

    float get() const { return prev_output; }

    void reset() {
        prev_input = 0;
        prev_output = 0;
        initialized = false;
    }
};

// =============================================================================
// 3-Axis Filter Set (for accelerometer/gyroscope)
// =============================================================================

class Vector3Filter {
private:
    EMAFilter filter_x;
    EMAFilter filter_y;
    EMAFilter filter_z;

public:
    Vector3Filter(float alpha = 0.3f)
        : filter_x(alpha), filter_y(alpha), filter_z(alpha) {}

    void update(float& x, float& y, float& z) {
        x = filter_x.update(x);
        y = filter_y.update(y);
        z = filter_z.update(z);
    }

    void reset() {
        filter_x.reset();
        filter_y.reset();
        filter_z.reset();
    }

    void setAlpha(float alpha) {
        filter_x.setAlpha(alpha);
        filter_y.setAlpha(alpha);
        filter_z.setAlpha(alpha);
    }
};

#endif // FILTERS_H
