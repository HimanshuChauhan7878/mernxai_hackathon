"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FileDown, BarChart3, BarChart2, Clock, MemoryStickIcon as Memory } from "lucide-react"
import {
  Chart,
  ChartContainer,
  ChartBar,
  ChartTitle,
  ChartXAxis,
  ChartYAxis,
  ChartTooltip,
  ChartGrid,
  ChartLegend,
} from "@/components/ui/chart"

interface BenchmarkResultsProps {
  mode: "beginner" | "nerds"
}

export function BenchmarkResults({ mode }: BenchmarkResultsProps) {
  const [selectedModel, setSelectedModel] = useState<string>("all")

  // Mock data for demonstration
  const benchmarkData = [
    {
      model: "ResNet50.pt",
      accuracy: 0.92,
      inferenceTime: 42.3,
      memoryUsage: 1.2,
      date: "2023-05-15",
    },
    {
      model: "MobileNetV2.h5",
      accuracy: 0.87,
      inferenceTime: 26.8,
      memoryUsage: 0.6,
      date: "2023-05-16",
    },
    {
      model: "EfficientNet.onnx",
      accuracy: 0.94,
      inferenceTime: 36.7,
      memoryUsage: 0.9,
      date: "2023-05-17",
    },
  ]

  const accuracyData = benchmarkData.map((item) => ({
    name: item.model,
    value: item.accuracy * 100,
  }))

  const inferenceData = benchmarkData.map((item) => ({
    name: item.model,
    value: item.inferenceTime,
  }))

  const memoryData = benchmarkData.map((item) => ({
    name: item.model,
    value: item.memoryUsage,
  }))

  const handleExportPDF = () => {
    console.log("Exporting benchmark report as PDF")
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-lg font-medium">Benchmark Results</h3>
          <p className="text-sm text-gray-500">
            {mode === "beginner"
              ? "View simplified performance metrics for your models"
              : "Detailed performance analysis and comparison metrics"}
          </p>
        </div>
        <Button onClick={handleExportPDF} className="bg-emerald-600 hover:bg-emerald-700">
          <FileDown className="mr-2 h-4 w-4" />
          Export Report
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center gap-2 text-center">
              <BarChart3 className="h-8 w-8 text-emerald-600" />
              <h3 className="font-medium">Accuracy</h3>
              <div className="text-3xl font-bold">{mode === "beginner" ? "91%" : "91.0%"}</div>
              <p className="text-xs text-gray-500">Average across all models</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center gap-2 text-center">
              <Clock className="h-8 w-8 text-emerald-600" />
              <h3 className="font-medium">Inference Time</h3>
              <div className="text-3xl font-bold">{mode === "beginner" ? "35ms" : "35.0ms"}</div>
              <p className="text-xs text-gray-500">Average across all models</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center gap-2 text-center">
              <Memory className="h-8 w-8 text-emerald-600" />
              <h3 className="font-medium">Memory Usage</h3>
              <div className="text-3xl font-bold">{mode === "beginner" ? "0.9GB" : "0.9GB"}</div>
              <p className="text-xs text-gray-500">Average across all models</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {mode === "beginner" ? (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <h3 className="font-medium">Model Performance</h3>
              <div className="h-[300px] w-full">
                <ChartContainer>
                  <ChartTitle>Model Accuracy (%)</ChartTitle>
                  <Chart>
                    <ChartGrid />
                    <ChartBar data={accuracyData} yAxisKey="value" xAxisKey="name" />
                    <ChartXAxis />
                    <ChartYAxis />
                    <ChartTooltip />
                  </Chart>
                </ChartContainer>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Tabs defaultValue="accuracy" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="inference">Inference Time</TabsTrigger>
            <TabsTrigger value="memory">Memory Usage</TabsTrigger>
          </TabsList>
          <TabsContent value="accuracy" className="space-y-4 pt-4">
            <Card>
              <CardContent className="pt-6">
                <div className="h-[350px] w-full">
                  <ChartContainer>
                    <ChartTitle>Model Accuracy (%)</ChartTitle>
                    <ChartLegend />
                    <Chart>
                      <ChartGrid />
                      <ChartBar data={accuracyData} yAxisKey="value" xAxisKey="name" />
                      <ChartXAxis />
                      <ChartYAxis />
                      <ChartTooltip />
                    </Chart>
                  </ChartContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="inference" className="space-y-4 pt-4">
            <Card>
              <CardContent className="pt-6">
                <div className="h-[350px] w-full">
                  <ChartContainer>
                    <ChartTitle>Inference Time (ms)</ChartTitle>
                    <ChartLegend />
                    <Chart>
                      <ChartGrid />
                      <ChartBar data={inferenceData} yAxisKey="value" xAxisKey="name" />
                      <ChartXAxis />
                      <ChartYAxis />
                      <ChartTooltip />
                    </Chart>
                  </ChartContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="memory" className="space-y-4 pt-4">
            <Card>
              <CardContent className="pt-6">
                <div className="h-[350px] w-full">
                  <ChartContainer>
                    <ChartTitle>Memory Usage (GB)</ChartTitle>
                    <ChartLegend />
                    <Chart>
                      <ChartGrid />
                      <ChartBar data={memoryData} yAxisKey="value" xAxisKey="name" />
                      <ChartXAxis />
                      <ChartYAxis />
                      <ChartTooltip />
                    </Chart>
                  </ChartContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">Detailed Results</h3>
              <div className="flex items-center gap-2">
                <label htmlFor="model-select" className="text-sm">
                  Filter:
                </label>
                <select
                  id="model-select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="rounded-md border border-gray-300 px-2 py-1 text-sm"
                >
                  <option value="all">All Models</option>
                  <option value="ResNet50.pt">ResNet50.pt</option>
                  <option value="MobileNetV2.h5">MobileNetV2.h5</option>
                  <option value="EfficientNet.onnx">EfficientNet.onnx</option>
                </select>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b">
                    <th className="py-2 text-left font-medium">Model</th>
                    <th className="py-2 text-left font-medium">Accuracy</th>
                    <th className="py-2 text-left font-medium">Inference Time</th>
                    <th className="py-2 text-left font-medium">Memory Usage</th>
                    {mode === "nerds" && <th className="py-2 text-left font-medium">Date</th>}
                    <th className="py-2 text-left font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkData
                    .filter((item) => selectedModel === "all" || item.model === selectedModel)
                    .map((item, index) => (
                      <tr key={index} className="border-b">
                        <td className="py-2">{item.model}</td>
                        <td className="py-2">{(item.accuracy * 100).toFixed(mode === "beginner" ? 0 : 1)}%</td>
                        <td className="py-2">{item.inferenceTime.toFixed(mode === "beginner" ? 0 : 1)}ms</td>
                        <td className="py-2">{item.memoryUsage.toFixed(1)}GB</td>
                        {mode === "nerds" && <td className="py-2">{item.date}</td>}
                        <td className="py-2">
                          <Button variant="outline" size="sm">
                            <BarChart2 className="mr-1 h-3 w-3" />
                            Details
                          </Button>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 