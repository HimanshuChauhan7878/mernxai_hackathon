import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { BarChart3, FileUp, BarChart2, Clock, MemoryStickIcon as Memory, ArrowRight } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <header className="border-b">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-emerald-600" />
            <span className="text-xl font-bold">BenchForge</span>
          </div>
          <nav className="flex items-center gap-6">
            <Link href="/" className="font-medium">
              Home
            </Link>
            <Link href="/dashboard" className="font-medium">
              Dashboard
            </Link>
            <Link href="/about" className="font-medium text-emerald-600">
              About
            </Link>
            <Button variant="default" className="bg-emerald-600 hover:bg-emerald-700">
              Get Started
            </Button>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <section className="py-12 md:py-16 lg:py-20">
          <div className="container px-4 md:px-6">
            <div className="mx-auto max-w-3xl text-center">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">About BenchForge</h1>
              <p className="mt-4 text-gray-500 md:text-xl/relaxed">
                BenchForge is a comprehensive AI model benchmarking tool designed to help developers evaluate and
                compare the performance of their machine learning models.
              </p>
            </div>
          </div>
        </section>

        <section className="py-12 md:py-16 lg:py-20 bg-gray-50">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-3 lg:gap-12">
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <FileUp className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="text-xl font-bold">Easy Model Upload</h3>
                <p className="text-gray-500">
                  Upload your AI models in various formats including PyTorch (.pt), Keras/TensorFlow (.h5), and ONNX
                  (.onnx) with just a few clicks.
                </p>
              </div>
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <BarChart2 className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="text-xl font-bold">Comprehensive Metrics</h3>
                <p className="text-gray-500">
                  Measure key performance indicators including accuracy, inference time, and memory usage to get a
                  complete picture of your model's performance.
                </p>
              </div>
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <FileUp className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="text-xl font-bold">Detailed Reports</h3>
                <p className="text-gray-500">
                  Generate comprehensive PDF reports with visualizations and metrics that you can share with your team
                  or include in your documentation.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 md:py-16 lg:py-20">
          <div className="container px-4 md:px-6">
            <div className="mx-auto max-w-3xl text-center">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl">Key Features</h2>
              <p className="mt-4 text-gray-500 md:text-xl/relaxed">
                BenchForge offers a range of features to help you evaluate and optimize your AI models.
              </p>
            </div>
            <div className="mx-auto mt-8 grid max-w-5xl gap-6 md:grid-cols-2 lg:grid-cols-3">
              <Card className="flex flex-col items-center p-6 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <BarChart3 className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="mt-4 text-xl font-bold">Two User Modes</h3>
                <p className="mt-2 text-gray-500">
                  Choose between Beginner and Nerds mode to get the level of detail that matches your expertise.
                </p>
              </Card>
              <Card className="flex flex-col items-center p-6 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <Clock className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="mt-4 text-xl font-bold">Performance Tracking</h3>
                <p className="mt-2 text-gray-500">
                  Track your model's performance over time and identify areas for optimization.
                </p>
              </Card>
              <Card className="flex flex-col items-center p-6 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <Memory className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="mt-4 text-xl font-bold">Resource Monitoring</h3>
                <p className="mt-2 text-gray-500">
                  Monitor CPU, GPU, and memory usage to understand your model's resource requirements.
                </p>
              </Card>
              <Card className="flex flex-col items-center p-6 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-6 w-6 text-emerald-600"
                  >
                    <path d="M12 2v8" />
                    <path d="m4.93 10.93 1.41 1.41" />
                    <path d="M2 18h2" />
                    <path d="M20 18h2" />
                    <path d="m19.07 10.93-1.41 1.41" />
                    <path d="M22 22H2" />
                    <path d="m16 6-4 4-4-4" />
                    <path d="M16 18a4 4 0 0 0-8 0" />
                  </svg>
                </div>
                <h3 className="mt-4 text-xl font-bold">Advanced Visualizations</h3>
                <p className="mt-2 text-gray-500">
                  Visualize your model's performance with interactive charts and graphs.
                </p>
              </Card>
              <Card className="flex flex-col items-center p-6 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-6 w-6 text-emerald-600"
                  >
                    <rect width="18" height="18" x="3" y="3" rx="2" />
                    <path d="M3 9h18" />
                    <path d="M9 21V9" />
                  </svg>
                </div>
                <h3 className="mt-4 text-xl font-bold">Model Comparison</h3>
                <p className="mt-2 text-gray-500">
                  Compare multiple models side by side to identify the best one for your use case.
                </p>
              </Card>
              <Card className="flex flex-col items-center p-6 text-center">
                <div className="rounded-full bg-emerald-100 p-3">
                  <FileUp className="h-6 w-6 text-emerald-600" />
                </div>
                <h3 className="mt-4 text-xl font-bold">Export Capabilities</h3>
                <p className="mt-2 text-gray-500">
                  Export your benchmark results as PDF reports for sharing and documentation.
                </p>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-12 md:py-16 lg:py-20 bg-gray-50">
          <div className="container px-4 md:px-6">
            <div className="mx-auto max-w-3xl text-center">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl">Get Started Today</h2>
              <p className="mt-4 text-gray-500 md:text-xl/relaxed">
                Start benchmarking your AI models with BenchForge and gain valuable insights into their performance.
              </p>
              <div className="mt-8 flex flex-col gap-2 min-[400px]:flex-row justify-center">
                <Link href="/dashboard">
                  <Button className="bg-emerald-600 hover:bg-emerald-700">
                    Go to Dashboard
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/">
                  <Button variant="outline">Learn More</Button>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t bg-gray-50">
        <div className="container flex flex-col gap-2 py-6 md:flex-row md:items-center md:justify-between md:py-8">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-emerald-600" />
            <span className="text-lg font-semibold">BenchForge</span>
          </div>
          <p className="text-xs text-gray-500 md:text-sm">© 2025 BenchForge. All rights reserved.</p>
          <div className="flex gap-4 text-xs text-gray-500 md:text-sm">
            <Link href="#" className="hover:underline">
              Terms
            </Link>
            <Link href="#" className="hover:underline">
              Privacy
            </Link>
            <Link href="#" className="hover:underline">
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}

