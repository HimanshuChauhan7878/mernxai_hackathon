import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, BarChart3, FileUp, FileDown, Zap } from "lucide-react"

export default function Home() {
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
            <Link href="/about" className="font-medium">
              About
            </Link>
            <Button variant="default" className="bg-emerald-600 hover:bg-emerald-700">
              Get Started
            </Button>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <section className="py-20 bg-gradient-to-b from-white to-gray-50">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 items-center">
              <div className="space-y-4">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                  AI Model Benchmarking Made Simple
                </h1>
                <p className="text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  BenchForge helps you evaluate and compare AI models with comprehensive performance metrics. Choose
                  between Beginner and Nerds mode to get the insights you need.
                </p>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Link href="/dashboard">
                    <Button className="bg-emerald-600 hover:bg-emerald-700">
                      Start Benchmarking
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                  <Link href="/about">
                    <Button variant="outline">Learn More</Button>
                  </Link>
                </div>
              </div>
              <div className="mx-auto lg:mx-0 rounded-lg overflow-hidden border bg-white shadow-lg">
                <div className="p-6 bg-white">
                  <div className="grid gap-6">
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-emerald-100 p-2">
                        <FileUp className="h-5 w-5 text-emerald-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Upload Models</h3>
                        <p className="text-sm text-gray-500">Support for .pt, .h5, .onnx formats</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-emerald-100 p-2">
                        <Zap className="h-5 w-5 text-emerald-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Run Benchmarks</h3>
                        <p className="text-sm text-gray-500">Measure accuracy, inference time, and memory usage</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-emerald-100 p-2">
                        <BarChart3 className="h-5 w-5 text-emerald-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Visualize Results</h3>
                        <p className="text-sm text-gray-500">Simple or advanced visualizations based on your mode</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-emerald-100 p-2">
                        <FileDown className="h-5 w-5 text-emerald-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Export Reports</h3>
                        <p className="text-sm text-gray-500">Download comprehensive PDF reports</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 md:py-16 lg:py-20 bg-white">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Choose Your Mode</h2>
                <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  BenchForge offers two user modes to match your expertise level
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-2 lg:gap-12">
              <div className="rounded-lg border bg-white p-6 shadow-sm">
                <div className="flex flex-col items-center space-y-4 text-center">
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
                      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                      <circle cx="9" cy="7" r="4" />
                      <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold">Beginner Mode</h3>
                  <p className="text-gray-500">
                    Simplified interface with easy-to-understand metrics and visualizations. Perfect for those new to AI
                    model benchmarking.
                  </p>
                  <ul className="text-sm text-gray-500 space-y-2">
                    <li>• Simplified performance summaries</li>
                    <li>• Basic visualizations</li>
                    <li>• Guided benchmarking process</li>
                    <li>• Easy-to-read reports</li>
                  </ul>
                </div>
              </div>
              <div className="rounded-lg border bg-white p-6 shadow-sm">
                <div className="flex flex-col items-center space-y-4 text-center">
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
                  <h3 className="text-xl font-bold">Nerds Mode</h3>
                  <p className="text-gray-500">
                    Advanced interface with detailed metrics, comprehensive visualizations, and in-depth analysis tools
                    for AI experts.
                  </p>
                  <ul className="text-sm text-gray-500 space-y-2">
                    <li>• Detailed performance metrics</li>
                    <li>• Advanced charts and graphs</li>
                    <li>• Model comparison tools</li>
                    <li>• Customizable benchmarking parameters</li>
                  </ul>
                </div>
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

