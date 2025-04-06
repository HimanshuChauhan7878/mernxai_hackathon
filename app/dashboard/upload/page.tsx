"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { FileUp, ArrowLeft } from "lucide-react"
import { ModelUploader } from "@/components/model-uploader"

export default function UploadPage() {
  const [modelType, setModelType] = useState<string>("")

  return (
    <div className="container mx-auto max-w-5xl py-6">
      <div className="mb-6 flex items-center gap-2">
        <Link href="/dashboard">
          <Button variant="outline" size="icon">
            <ArrowLeft className="h-4 w-4" />
            <span className="sr-only">Back</span>
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Upload Model</h1>
          <p className="text-gray-500">Upload your AI models for benchmarking</p>
        </div>
      </div>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Upload Model</CardTitle>
            <CardDescription>Upload your AI model file (.pt, .h5, .onnx)</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelUploader />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Details</CardTitle>
            <CardDescription>Provide additional information about your model</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label htmlFor="model-name">Model Name</Label>
                <Input id="model-name" placeholder="e.g., ResNet50" />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="model-type">Model Type</Label>
                <Select value={modelType} onValueChange={setModelType}>
                  <SelectTrigger id="model-type">
                    <SelectValue placeholder="Select model type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="detection">Object Detection</SelectItem>
                    <SelectItem value="segmentation">Segmentation</SelectItem>
                    <SelectItem value="nlp">Natural Language Processing</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="framework">Framework</Label>
                <Select>
                  <SelectTrigger id="framework">
                    <SelectValue placeholder="Select framework" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pytorch">PyTorch</SelectItem>
                    <SelectItem value="tensorflow">TensorFlow/Keras</SelectItem>
                    <SelectItem value="onnx">ONNX</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="version">Version</Label>
                <Input id="version" placeholder="e.g., 1.0.0" />
              </div>
              <div className="col-span-2 grid gap-2">
                <Label htmlFor="description">Description</Label>
                <Textarea id="description" placeholder="Provide a brief description of your model" rows={3} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="basic">Basic Configuration</TabsTrigger>
            <TabsTrigger value="advanced">Advanced Configuration</TabsTrigger>
          </TabsList>
          <TabsContent value="basic" className="space-y-4 pt-4">
            <Card>
              <CardHeader>
                <CardTitle>Benchmark Configuration</CardTitle>
                <CardDescription>Configure basic benchmarking parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid gap-2">
                    <Label htmlFor="dataset">Dataset</Label>
                    <Select>
                      <SelectTrigger id="dataset">
                        <SelectValue placeholder="Select dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="imagenet">ImageNet</SelectItem>
                        <SelectItem value="coco">COCO</SelectItem>
                        <SelectItem value="cifar10">CIFAR-10</SelectItem>
                        <SelectItem value="custom">Custom</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="batch-size">Batch Size</Label>
                    <Input id="batch-size" type="number" defaultValue={1} min={1} />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="iterations">Iterations</Label>
                    <Input id="iterations" type="number" defaultValue={100} min={1} />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="device">Device</Label>
                    <Select>
                      <SelectTrigger id="device">
                        <SelectValue placeholder="Select device" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="cpu">CPU</SelectItem>
                        <SelectItem value="gpu">GPU</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="advanced" className="space-y-4 pt-4">
            <Card>
              <CardHeader>
                <CardTitle>Advanced Configuration</CardTitle>
                <CardDescription>Configure advanced benchmarking parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid gap-2">
                    <Label htmlFor="precision">Precision</Label>
                    <Select>
                      <SelectTrigger id="precision">
                        <SelectValue placeholder="Select precision" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="fp32">FP32</SelectItem>
                        <SelectItem value="fp16">FP16</SelectItem>
                        <SelectItem value="int8">INT8</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="warmup">Warmup Iterations</Label>
                    <Input id="warmup" type="number" defaultValue={10} min={0} />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="threads">Number of Threads</Label>
                    <Input id="threads" type="number" defaultValue={4} min={1} />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="timeout">Timeout (seconds)</Label>
                    <Input id="timeout" type="number" defaultValue={300} min={1} />
                  </div>
                  <div className="col-span-2 grid gap-2">
                    <Label htmlFor="custom-params">Custom Parameters</Label>
                    <Textarea id="custom-params" placeholder="Enter custom parameters in JSON format" rows={3} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="flex justify-end gap-2">
          <Button variant="outline">Cancel</Button>
          <Button className="bg-emerald-600 hover:bg-emerald-700">
            <FileUp className="mr-2 h-4 w-4" />
            Upload and Benchmark
          </Button>
        </div>
      </div>
    </div>
  )
}

