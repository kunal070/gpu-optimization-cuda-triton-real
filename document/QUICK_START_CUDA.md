# Quick Start: Running with CUDA on Windows

## 🚀 Fastest Way to Get Started

### Step 1: Open x64 Native Tools Command Prompt

1. Press `Win + S`
2. Search for: **"x64 Native Tools Command Prompt for VS 2022"**
3. Open it
4. Navigate to your project:
   ```cmd
   cd C:\Users\kunal\OneDrive\Desktop\yash-luli
   ```

### Step 2: Build CUDA Extension

```cmd
python setup.py build_ext --inplace
```

⏱️ **Takes 5-10 minutes** (first time only)

### Step 3: Test It Works

```cmd
python test_cuda.py
```

Should see: `All CUDA extension tests passed!`

### Step 4: Run Benchmarks

```cmd
python benchmarks/comprehensive_benchmark.py
```

This will compare **PyTorch vs CUDA** performance!

---

## 📋 Complete Command Sequence

Copy and paste these commands one by one in **x64 Native Tools Command Prompt**:

```cmd
REM 1. Navigate to project
cd C:\Users\kunal\OneDrive\Desktop\yash-luli

REM 2. Set CUDA path (if needed)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

REM 3. Build extension
python setup.py build_ext --inplace

REM 4. Test it
python test_cuda.py

REM 5. Run full tests
python tests/test_basic_kernels.py

REM 6. Run benchmarks
python benchmarks/comprehensive_benchmark.py
```

---

## ⚠️ Important Notes

1. **MUST use x64 Native Tools Command Prompt** - Regular CMD/PowerShell won't work
2. **First build takes 5-10 minutes** - This is normal
3. **Triton won't work on Windows** - But CUDA gives similar performance
4. **Results saved to** `benchmark_results.json`

---

## 🐛 Troubleshooting

### "CUDA not found"
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
```

### "cl.exe not found"
- Make sure you're using **x64 Native Tools Command Prompt**
- Install Visual Studio Build Tools

### Build fails
- Check you're in x64 Native Tools Command Prompt
- Verify CUDA_PATH is set
- Try: `pip install ninja`

---

## ✅ Success Checklist

- [ ] Opened x64 Native Tools Command Prompt
- [ ] Built extension: `python setup.py build_ext --inplace`
- [ ] Tested: `python test_cuda.py` (all tests pass)
- [ ] Ran benchmarks: `python benchmarks/comprehensive_benchmark.py`
- [ ] Results in `benchmark_results.json`

---

**That's it! You're running CUDA kernels on Windows! 🎉**

