#!/bin/bash

echo "================================================================================"
echo "SERVER RESOURCE CHECK"
echo "================================================================================"
echo ""

# ==============================================================================
# DISK SPACE
# ==============================================================================
echo "================================================================================"
echo "DISK SPACE ANALYSIS"
echo "================================================================================"
echo ""

echo "Overall Disk Usage:"
echo "-------------------"
df -h | head -1
df -h | grep -E '^/dev/'
echo ""

echo "Your Home Directory Usage:"
echo "--------------------------"
du -sh ~ 2>/dev/null
echo ""

echo "Your Project Directory Usage:"
echo "------------------------------"
if [ -d ~/wildfire_burn_scar_mapping ]; then
    du -sh ~/wildfire_burn_scar_mapping 2>/dev/null
    echo ""
    echo "Breakdown by folder:"
    du -h --max-depth=2 ~/wildfire_burn_scar_mapping 2>/dev/null | sort -h | tail -20
else
    echo "Project directory not found at ~/wildfire_burn_scar_mapping"
fi
echo ""

echo "Available Space on Home Partition:"
echo "-----------------------------------"
df -h ~ | tail -1 | awk '{print "Total: "$2"  Used: "$3"  Available: "$4"  Use%: "$5}'
echo ""

# ==============================================================================
# STORAGE QUOTA (if applicable)
# ==============================================================================
echo "================================================================================"
echo "STORAGE QUOTA CHECK"
echo "================================================================================"
echo ""

if command -v quota &> /dev/null; then
    echo "Your Quota:"
    quota -s 2>/dev/null || echo "No quota set or command not available"
else
    echo "Quota command not available (you may have unlimited storage)"
fi
echo ""

# ==============================================================================
# LARGE FILES
# ==============================================================================
echo "================================================================================"
echo "LARGEST FILES IN PROJECT (Top 20)"
echo "================================================================================"
echo ""

if [ -d ~/wildfire_burn_scar_mapping ]; then
    find ~/wildfire_burn_scar_mapping -type f -exec du -h {} + 2>/dev/null | sort -h | tail -20
else
    echo "Project directory not found"
fi
echo ""

# ==============================================================================
# MEMORY
# ==============================================================================
echo "================================================================================"
echo "MEMORY (RAM)"
echo "================================================================================"
echo ""

if command -v free &> /dev/null; then
    free -h
else
    echo "Memory info not available"
fi
echo ""

# ==============================================================================
# CPU
# ==============================================================================
echo "================================================================================"
echo "CPU INFO"
echo "================================================================================"
echo ""

if [ -f /proc/cpuinfo ]; then
    echo "CPU Cores: $(grep -c processor /proc/cpuinfo)"
    echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
else
    echo "CPU info not available"
fi
echo ""

# ==============================================================================
# NETWORK BANDWIDTH (if available)
# ==============================================================================
echo "================================================================================"
echo "NETWORK INFO"
echo "================================================================================"
echo ""

if command -v speedtest-cli &> /dev/null; then
    echo "Running speed test (this may take 30-60 seconds)..."
    speedtest-cli --simple
else
    echo "speedtest-cli not installed (skip network test)"
    echo "Install with: pip install speedtest-cli"
fi
echo ""

# ==============================================================================
# SUMMARY & RECOMMENDATIONS
# ==============================================================================
echo "================================================================================"
echo "SUMMARY & RECOMMENDATIONS"
echo "================================================================================"
echo ""

# Extract available space in GB
AVAIL_SPACE=$(df ~ | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_SPACE / 1024 / 1024))

echo "Available Storage: ~${AVAIL_GB} GB"
echo ""

if [ $AVAIL_GB -gt 3000 ]; then
    echo "✅ EXCELLENT: You have enough space for full MTBS download (~1-3 TB)"
    echo "   → Option 2 (Full Download) is feasible"
elif [ $AVAIL_GB -gt 1000 ]; then
    echo "✅ GOOD: You have enough space for full MTBS download (~1-3 TB)"
    echo "   → Option 2 (Full Download) is feasible, but monitor closely"
elif [ $AVAIL_GB -gt 100 ]; then
    echo "⚠️  LIMITED: You have space for ~100-500 fire samples"
    echo "   → Hybrid approach recommended (sample MTBS + GEE for rest)"
    echo "   → Each fire: ~50-150 MB, you can fit ~${AVAIL_GB}0-2000 fires"
elif [ $AVAIL_GB -gt 30 ]; then
    echo "⚠️  TIGHT: You have space for ~20-100 fire samples"
    echo "   → Hybrid approach with small validation set"
    echo "   → Use free academic GEE for bulk processing"
else
    echo "❌ INSUFFICIENT: Not enough space for MTBS downloads"
    echo "   → Must use free academic GEE approach"
    echo "   → Or request additional storage from admin"
fi
echo ""

echo "Storage needed for different approaches:"
echo "  • Full MTBS download:        1,000 - 3,000 GB"
echo "  • Hybrid (200 samples):         10 -    30 GB"
echo "  • GEE only (cache):              5 -    10 GB"
echo ""

echo "================================================================================"
echo "RESOURCE CHECK COMPLETE"
echo "================================================================================"
