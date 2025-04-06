"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { 
  BarChart3, 
  FileUp, 
  FileDown, 
  Home, 
  Settings, 
  User, 
  Brain, 
  Layers, 
  Cpu, 
  TrendingUp,
  HelpCircle,
  TerminalSquare 
} from "lucide-react"
import { ModelUploader } from "@/components/model-uploader"
import { ModelList } from "@/components/model-list"
import { BenchmarkResults } from "@/components/benchmark-results"
import { toast } from "sonner"

export default function Dashboard() {
  const [mode, setMode] = useState<"beginner" | "nerds">("beginner")
  const [activeTab, setActiveTab] = useState("overview")
  const router = useRouter()

  const handleUploadNewModel = () => {
    setActiveTab("models");
    toast.success("Switched to Models tab");
  }

  const handleRunBenchmark = () => {
    setActiveTab("benchmarks");
    toast.success("Switched to Benchmarks tab");
  }

  const handleGenerateReport = () => {
    router.push("/dashboard/reports");
    toast.success("Navigating to Reports page");
  }

  const handleModeChange = (checked: boolean) => {
    const newMode = checked ? "nerds" : "beginner";
    setMode(newMode);
    toast.success(`Switched to ${newMode === "beginner" ? "Beginner" : "Nerds"} Mode`);
  }

  const handleShowHelp = () => {
    toast.info(
      mode === "beginner" 
        ? "Beginner Mode: Simplified interface for easy benchmarking" 
        : "Nerds Mode: Advanced features and detailed metrics for power users"
    );
  }

  // Render content based on mode
  const renderBeginnerMode = () => {
    return (
      <div className="flex flex-col gap-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-gray-500">
            Welcome to BenchForge! Upload your AI models and start benchmarking.
          </p>
          <Button variant="link" className="p-0 text-blue-500" onClick={handleShowHelp}>
            <HelpCircle className="h-4 w-4 mr-1" />
            How to use BenchForge
          </Button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Quick Start Guide</CardTitle>
            <CardDescription>Follow these steps to benchmark your model</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-start space-x-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                1
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-medium">Upload Your Model</h3>
                <p className="text-sm text-gray-500">Select your model file (.pt, .h5, .onnx) to upload</p>
                <Button 
                  className="mt-2 bg-emerald-600 hover:bg-emerald-700"
                  onClick={handleUploadNewModel}
                >
                  <FileUp className="mr-2 h-4 w-4" />
                  Upload Model
                </Button>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                2
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-medium">Run Benchmarks</h3>
                <p className="text-sm text-gray-500">Test your model's performance</p>
                <Button 
                  className="mt-2"
                  variant="outline"
                  onClick={handleRunBenchmark}
                >
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Run Benchmark
                </Button>
              </div>
            </div>
            
            <div className="flex items-start space-x-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                3
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-medium">Generate Report</h3>
                <p className="text-sm text-gray-500">Create a shareable performance report</p>
                <Button 
                  className="mt-2"
                  variant="outline"
                  onClick={handleGenerateReport}
                >
                  <FileDown className="mr-2 h-4 w-4" />
                  Generate Report
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Your Models</CardTitle>
          </CardHeader>
          <CardContent>
            <ModelList mode={mode} />
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderNerdsMode = () => {
    return (
      <div className="flex flex-col gap-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-gray-500">
            Advanced benchmarking dashboard with detailed metrics and visualizations.
          </p>
          <div className="flex gap-2 mt-2">
            <Button variant="outline" size="sm" onClick={() => toast.info("Terminal access coming soon!")}>
              <TerminalSquare className="h-4 w-4 mr-1" />
              Terminal
            </Button>
            <Button variant="outline" size="sm" onClick={() => toast.info("API documentation coming soon!")}>
              <Cpu className="h-4 w-4 mr-1" />
              API Docs
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="benchmarks">Benchmarks</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="space-y-4 pt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Models</CardTitle>
                  <FileUp className="h-4 w-4 text-gray-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">3</div>
                  <p className="text-xs text-gray-500">+1 from last week</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Benchmarks Run</CardTitle>
                  <BarChart3 className="h-4 w-4 text-gray-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">12</div>
                  <p className="text-xs text-gray-500">+4 from last week</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Reports Generated</CardTitle>
                  <FileDown className="h-4 w-4 text-gray-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">5</div>
                  <p className="text-xs text-gray-500">+2 from last week</p>
                </CardContent>
              </Card>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle>Recent Models</CardTitle>
                  <CardDescription>Your recently uploaded AI models</CardDescription>
                </CardHeader>
                <CardContent>
                  <ModelList mode={mode} />
                </CardContent>
              </Card>
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                  <CardDescription>Common tasks and shortcuts</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-2">
                  <Button 
                    className="w-full justify-start bg-emerald-600 hover:bg-emerald-700"
                    onClick={handleUploadNewModel}
                  >
                    <FileUp className="mr-2 h-4 w-4" />
                    Upload New Model
                  </Button>
                  <Button 
                    variant="outline" 
                    className="w-full justify-start"
                    onClick={handleRunBenchmark}
                  >
                    <BarChart3 className="mr-2 h-4 w-4" />
                    Run Benchmark
                  </Button>
                  <Button 
                    variant="outline" 
                    className="w-full justify-start"
                    onClick={handleGenerateReport}
                  >
                    <FileDown className="mr-2 h-4 w-4" />
                    Generate Report
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="models" className="space-y-4 pt-4">
            <Card>
              <CardHeader>
                <CardTitle>Upload Model</CardTitle>
                <CardDescription>Upload your AI models (.pt, .h5, .onnx) for benchmarking</CardDescription>
              </CardHeader>
              <CardContent>
                <ModelUploader />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Your Models</CardTitle>
                <CardDescription>Manage your uploaded AI models</CardDescription>
              </CardHeader>
              <CardContent>
                <ModelList mode={mode} />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="benchmarks" className="space-y-4 pt-4">
            <Card>
              <CardHeader>
                <CardTitle>Benchmark Results</CardTitle>
                <CardDescription>View performance metrics for your AI models</CardDescription>
              </CardHeader>
              <CardContent>
                <BenchmarkResults mode={mode} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    );
  };

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-emerald-600" />
            <span className="text-xl font-bold">BenchForge</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center space-x-2">
              <Switch
                id="mode-switch"
                checked={mode === "nerds"}
                onCheckedChange={handleModeChange}
              />
              <Label htmlFor="mode-switch" className="flex items-center">
                {mode === "beginner" ? (
                  <>
                    <Brain className="h-4 w-4 mr-1 text-blue-500" />
                    <span>Beginner Mode</span>
                  </>
                ) : (
                  <>
                    <Layers className="h-4 w-4 mr-1 text-purple-500" />
                    <span>Nerds Mode</span>
                  </>
                )}
              </Label>
            </div>
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => toast.info("User profile coming soon!")}
            >
              <User className="h-4 w-4" />
              <span className="sr-only">User</span>
            </Button>
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => toast.info("Settings page coming soon!")}
            >
              <Settings className="h-4 w-4" />
              <span className="sr-only">Settings</span>
            </Button>
          </div>
        </div>
      </header>
      <div className="flex flex-1">
        <aside className="hidden w-64 border-r bg-gray-50 lg:block">
          <div className="flex h-full flex-col gap-2 p-4">
            <div className="px-2 py-2">
              <h2 className="text-lg font-semibold tracking-tight">Navigation</h2>
            </div>
            <div className="flex-1">
              <nav className="grid gap-1 px-2 group-[[data-collapsed=true]]:justify-center group-[[data-collapsed=true]]:px-2">
                <Link
                  href="/"
                  className="flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900"
                >
                  <Home className="h-4 w-4" />
                  <span>Home</span>
                </Link>
                <Link
                  href="/dashboard"
                  className="flex items-center gap-3 rounded-lg bg-gray-100 px-3 py-2 text-gray-900 transition-all hover:text-gray-900"
                >
                  <BarChart3 className="h-4 w-4" />
                  <span>Dashboard</span>
                </Link>
                <Link
                  href="/dashboard/upload"
                  className="flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900"
                >
                  <FileUp className="h-4 w-4" />
                  <span>Upload Models</span>
                </Link>
                <Link
                  href="/dashboard/reports"
                  className="flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900"
                >
                  <FileDown className="h-4 w-4" />
                  <span>Reports</span>
                </Link>
                {mode === "nerds" && (
                  <Link
                    href="#"
                    onClick={(e) => {
                      e.preventDefault();
                      toast.info("Advanced Analytics coming soon!");
                    }}
                    className="flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900"
                  >
                    <TrendingUp className="h-4 w-4" />
                    <span>Advanced Analytics</span>
                  </Link>
                )}
              </nav>
            </div>
          </div>
        </aside>
        <main className="flex-1 overflow-auto p-4 md:p-6">
          {mode === "beginner" ? renderBeginnerMode() : renderNerdsMode()}
        </main>
      </div>
    </div>
  )
}

