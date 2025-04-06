"use client"

import type React from "react"
import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { X, Activity, Zap, Clock, Cpu, Download, RefreshCw, Trash2, ArrowLeft } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js"
import { toast } from "sonner"

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface BenchmarkMetrics {
  inferenceTime: number;
  memoryUsage: number;
  throughput: number;
  accuracy: number;
}

type BenchmarkMode = "cpu" | "gpu" | "edge"

export function ModelUploader() {
  const [file, setFile] = useState<File | null>(null)
  const [benchmarking, setBenchmarking] = useState(false)
  const [metrics, setMetrics] = useState<BenchmarkMetrics | null>(null)
  const [mode, setMode] = useState<BenchmarkMode>("cpu")
  const [benchmarkHistory, setBenchmarkHistory] = useState<{ mode: BenchmarkMode; metrics: BenchmarkMetrics }[]>([])
  const [isExporting, setIsExporting] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      const maxSize = 500 * 1024 * 1024 // 500MB
      
      if (selectedFile.size > maxSize) {
        toast.error("File size exceeds 500MB limit")
        return
      }
      
      setFile(selectedFile)
      setMetrics(null)
      setBenchmarkHistory([])
      toast.success("Model file loaded successfully")
    }
  }

  const handleClear = () => {
    setFile(null)
    setMetrics(null)
    setBenchmarkHistory([])
    toast.success("Model cleared successfully")
  }

  const handleExportResults = () => {
    if (!benchmarkHistory.length) {
      toast.error("No benchmark results to export")
      return
    }

    setIsExporting(true)
    const data = {
      fileName: file?.name,
      benchmarkHistory: benchmarkHistory,
      timestamp: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `benchmark_results_${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    setIsExporting(false)
    toast.success("Results exported successfully")
  }

  const handleClearHistory = () => {
    setBenchmarkHistory([])
    toast.success("Benchmark history cleared")
  }

  const simulateBenchmark = () => {
    if (!file) {
      toast.error("Please select a model file first")
      return
    }

    setBenchmarking(true)
    
    // Simulate random benchmarking metrics based on mode
    const baseMetrics = {
      inferenceTime: Math.random() * 100 + 20,
      memoryUsage: Math.random() * 2000 + 500,
      throughput: Math.random() * 1000 + 500,
      accuracy: Math.random() * 0.1 + 0.9,
    }

    // Adjust metrics based on mode
    const modeMultipliers = {
      cpu: { inferenceTime: 1, memoryUsage: 1, throughput: 1 },
      gpu: { inferenceTime: 0.3, memoryUsage: 1.5, throughput: 3 },
      edge: { inferenceTime: 2, memoryUsage: 0.5, throughput: 0.5 },
    }

    const randomMetrics: BenchmarkMetrics = {
      inferenceTime: baseMetrics.inferenceTime * modeMultipliers[mode].inferenceTime,
      memoryUsage: baseMetrics.memoryUsage * modeMultipliers[mode].memoryUsage,
      throughput: baseMetrics.throughput * modeMultipliers[mode].throughput,
      accuracy: baseMetrics.accuracy,
    }

    // Simulate loading time
    setTimeout(() => {
      setMetrics(randomMetrics)
      setBenchmarkHistory(prev => [...prev, { mode, metrics: randomMetrics }])
      setBenchmarking(false)
      toast.success(`Benchmark completed in ${mode.toUpperCase()} mode`)
    }, 2000)
  }

  const chartData = {
    labels: benchmarkHistory.map((_, index) => `Run ${index + 1}`),
    datasets: [
      {
        label: "Inference Time (ms)",
        data: benchmarkHistory.map(h => h.metrics.inferenceTime),
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.5)",
        yAxisID: "y",
      },
      {
        label: "Throughput (inf/sec)",
        data: benchmarkHistory.map(h => h.metrics.throughput),
        borderColor: "rgb(234, 179, 8)",
        backgroundColor: "rgba(234, 179, 8, 0.5)",
        yAxisID: "y1",
      },
    ],
  }

  const chartOptions = {
    responsive: true,
    interaction: {
      mode: "index" as const,
      intersect: false,
    },
    scales: {
      y: {
        type: "linear" as const,
        display: true,
        position: "left" as const,
      },
      y1: {
        type: "linear" as const,
        display: true,
        position: "right" as const,
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 mb-6">
        <Link href="/">
          <Button variant="outline" size="sm" className="flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
        </Link>
      </div>

      <div className="grid w-full gap-1.5">
        <Label htmlFor="model">Model File</Label>
        <div className="flex items-center gap-2">
          <Input
            id="model"
            type="file"
            accept=".pt,.h5,.onnx"
            onChange={handleFileChange}
            className="flex-1"
          />
          {file && (
            <Button variant="outline" size="icon" onClick={handleClear}>
              <X className="h-4 w-4" />
              <span className="sr-only">Clear</span>
            </Button>
          )}
        </div>
        <p className="text-sm text-gray-500">Supported formats: .pt (PyTorch), .h5 (Keras/TensorFlow), .onnx (ONNX)</p>
      </div>

      {file && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">
                {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
              </span>
              <Button 
                variant="destructive" 
                size="sm" 
                onClick={handleClear}
                className="text-xs px-2"
              >
                <Trash2 className="h-3 w-3 mr-1" /> Delete Model
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <Select value={mode} onValueChange={(value: BenchmarkMode) => setMode(value)}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Select mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cpu">CPU Mode</SelectItem>
                  <SelectItem value="gpu">GPU Mode</SelectItem>
                  <SelectItem value="edge">Edge Mode</SelectItem>
                </SelectContent>
              </Select>
              <Button 
                onClick={simulateBenchmark}
                disabled={benchmarking}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Activity className="mr-2 h-4 w-4" />
                {benchmarking ? "Benchmarking..." : "Run Benchmark"}
              </Button>
            </div>
          </div>

          {benchmarking && (
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              <div className="h-4 bg-gray-200 rounded w-2/3"></div>
            </div>
          )}

          {metrics && (
            <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2">
                <Clock className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm font-medium">Inference Time</p>
                  <p className="text-sm text-gray-600">{metrics.inferenceTime.toFixed(2)}ms</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Cpu className="h-5 w-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium">Memory Usage</p>
                  <p className="text-sm text-gray-600">{metrics.memoryUsage.toFixed(0)}MB</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="h-5 w-5 text-yellow-600" />
                <div>
                  <p className="text-sm font-medium">Throughput</p>
                  <p className="text-sm text-gray-600">{metrics.throughput.toFixed(0)} inf/sec</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-purple-600" />
                <div>
                  <p className="text-sm font-medium">Accuracy</p>
                  <p className="text-sm text-gray-600">{(metrics.accuracy * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          )}

          {benchmarkHistory.length > 0 && (
            <div className="p-4 bg-white rounded-lg border">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Benchmark History</h3>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleExportResults}
                    disabled={isExporting}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export Results
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleClearHistory}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Clear History
                  </Button>
                </div>
              </div>
              <div className="h-[300px]">
                <Line options={chartOptions} data={chartData} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

