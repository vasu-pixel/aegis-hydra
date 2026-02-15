#!/bin/bash
# Fix CPU governor and priority for HFT performance

echo "=== AEGIS-HYDRA PERFORMANCE SETUP ==="

# 1. Set CPU governor to performance (prevents frequency scaling)
echo "Setting CPU governor to performance mode..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu > /dev/null 2>&1
done

# Check if it worked
GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
if [ "$GOVERNOR" = "performance" ]; then
    echo "✅ CPU governor set to performance"
else
    echo "⚠️  Could not set CPU governor (need sudo or not supported)"
    echo "   Current: $GOVERNOR"
fi

# 2. Disable CPU idle states (prevents deep sleep)
echo "Disabling CPU idle states..."
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 | sudo tee $state > /dev/null 2>&1
done

# 3. Set process priority
echo "Process priority will be set when daemon starts (os.nice(-20))"

# 4. Show current status
echo -e "\n=== CURRENT STATUS ==="
echo "CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
echo "CPU Freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 'N/A') Hz"
echo "Max Freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 'N/A') Hz"

echo -e "\n✅ Performance setup complete!"
echo "Run your HFT pipeline now: python3 -m aegis_hydra.tools.hft_pipe"
