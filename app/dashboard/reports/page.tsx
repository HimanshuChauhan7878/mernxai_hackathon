"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { BarChart3, FileDown, ArrowLeft, Search, Calendar, Filter } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function ReportsPage() {
  const [searchQuery, setSearchQuery] = useState("")

  // Mock data for demonstration
  const reports = [
    {
      id: 1,
      name: "ResNet50 Benchmark Report",
      model: "ResNet50.pt",
      date: "2025-04-02",
      size: "1.2 MB",
    },
    {
      id: 2,
      name: "MobileNetV2 Performance Analysis",
      model: "MobileNetV2.h5",
      date: "2025-04-01",
      size: "0.9 MB",
    },
    {
      id: 3,
      name: "Model Comparison Report",
      model: "Multiple",
      date: "2025-03-28",
      size: "2.4 MB",
    },
    {
      id: 4,
      name: "EfficientNet Benchmark",
      model: "EfficientNet.onnx",
      date: "2025-03-25",
      size: "1.1 MB",
    },
    {
      id: 5,
      name: "Weekly Performance Summary",
      model: "Multiple",
      date: "2025-03-21",
      size: "3.2 MB",
    },
  ]

  const filteredReports = reports.filter(
    (report) =>
      report.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      report.model.toLowerCase().includes(searchQuery.toLowerCase()),
  )

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
          <h1 className="text-2xl font-bold tracking-tight">Reports</h1>
          <p className="text-gray-500">View and download benchmark reports</p>
        </div>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Generate New Report</CardTitle>
          <CardDescription>Create a new benchmark report from your models</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Button className="w-full bg-emerald-600 hover:bg-emerald-700">
                <BarChart3 className="mr-2 h-4 w-4" />
                Single Model Report
              </Button>
            </div>
            <div>
              <Button className="w-full bg-emerald-600 hover:bg-emerald-700">
                <BarChart3 className="mr-2 h-4 w-4" />
                Comparison Report
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Your Reports</CardTitle>
          <CardDescription>View and download your benchmark reports</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex flex-col gap-4 sm:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500" />
              <Input
                placeholder="Search reports..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="icon">
                <Calendar className="h-4 w-4" />
                <span className="sr-only">Filter by date</span>
              </Button>
              <Button variant="outline" size="icon">
                <Filter className="h-4 w-4" />
                <span className="sr-only">Filter</span>
              </Button>
            </div>
          </div>

          {filteredReports.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
              <p className="text-sm text-gray-500">No reports found</p>
              <Button className="mt-4 bg-emerald-600 hover:bg-emerald-700">Generate Your First Report</Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Report Name</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredReports.map((report) => (
                    <TableRow key={report.id}>
                      <TableCell className="font-medium">{report.name}</TableCell>
                      <TableCell>{report.model}</TableCell>
                      <TableCell>{report.date}</TableCell>
                      <TableCell>{report.size}</TableCell>
                      <TableCell>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">
                            <BarChart3 className="mr-1 h-3 w-3" />
                            View
                          </Button>
                          <Button variant="outline" size="sm">
                            <FileDown className="mr-1 h-3 w-3" />
                            Download
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

