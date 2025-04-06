"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { BarChart3, Trash2, Info } from "lucide-react"
import { toast } from "sonner"

interface ModelListProps {
  mode: "beginner" | "nerds"
}

export function ModelList({ mode }: ModelListProps) {
  // Sample data for demo purposes
  const models = [
    { id: 1, name: "ResNet50.pt", size: "98.5 MB", type: "PyTorch", createdAt: "2023-04-15", accuracy: "93.2%" },
    { id: 2, name: "MobileNetV2.h5", size: "14.2 MB", type: "TensorFlow", createdAt: "2023-04-18", accuracy: "87.5%" },
    { id: 3, name: "YOLOv5.onnx", size: "245.7 MB", type: "ONNX", createdAt: "2023-04-20", accuracy: "95.8%" },
  ]

  const handleDelete = (id: number) => {
    toast.success(`Model ID ${id} deleted`)
  }

  const handleRunBenchmark = (id: number) => {
    toast.success(`Running benchmark for model ID ${id}`)
  }

  const handleViewDetails = (id: number) => {
    toast.info(`Viewing details for model ID ${id}`)
  }

  // Render different views based on mode
  if (mode === "beginner") {
    // Simplified view for beginners
    return (
      <div className="space-y-4">
        {models.map((model) => (
          <div key={model.id} className="flex justify-between items-center p-3 border rounded-md hover:bg-gray-50">
            <div>
              <h3 className="font-medium">{model.name}</h3>
              <p className="text-sm text-gray-500">Size: {model.size}</p>
            </div>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => handleRunBenchmark(model.id)}
              >
                <BarChart3 className="h-4 w-4 mr-1" />
                Benchmark
              </Button>
              <Button 
                variant="destructive" 
                size="sm"
                onClick={() => handleDelete(model.id)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        ))}
      </div>
    )
  } 
  
  // Advanced view for "nerds" mode
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-5 gap-4 px-3 py-2 font-medium text-sm text-gray-500">
        <div>Model Name</div>
        <div>Type</div>
        <div>Size</div>
        <div>Accuracy</div>
        <div className="text-right">Actions</div>
      </div>
      
      {models.map((model) => (
        <div key={model.id} className="grid grid-cols-5 gap-4 p-3 border rounded-md hover:bg-gray-50 items-center">
          <div>
            <h3 className="font-medium">{model.name}</h3>
            <p className="text-xs text-gray-500">Added {model.createdAt}</p>
          </div>
          <div>
            <Badge variant="outline">{model.type}</Badge>
          </div>
          <div>{model.size}</div>
          <div>{model.accuracy}</div>
          <div className="flex gap-2 justify-end">
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => handleViewDetails(model.id)}
            >
              <Info className="h-4 w-4" />
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleRunBenchmark(model.id)}
            >
              <BarChart3 className="h-4 w-4 mr-1" />
              Benchmark
            </Button>
            <Button 
              variant="destructive" 
              size="icon"
              onClick={() => handleDelete(model.id)}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  )
}

