"use client"

import * as React from "react"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

interface ChartProps {
  children: React.ReactNode
}

export function Chart({ children }: ChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={[]}>
        {children}
      </BarChart>
    </ResponsiveContainer>
  )
}

interface ChartBarProps {
  data: any[]
  xAxisKey: string
  yAxisKey: string
  fill?: string
}

export function ChartBar({ data, xAxisKey, yAxisKey, fill = "rgba(16, 185, 129, 0.8)" }: ChartBarProps) {
  return <Bar dataKey={yAxisKey} fill={fill} radius={4} />
}

interface ChartContainerProps {
  children: React.ReactNode
}

export function ChartContainer({ children }: ChartContainerProps) {
  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={[]} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          {children}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface ChartTitleProps {
  children: React.ReactNode
}

export function ChartTitle({ children }: ChartTitleProps) {
  return <h3 className="text-sm font-medium text-center mb-3">{children}</h3>
}

export function ChartXAxis() {
  return <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
}

export function ChartYAxis() {
  return <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
}

export function ChartTooltip() {
  return <Tooltip />
}

export function ChartGrid() {
  return <CartesianGrid strokeDasharray="3 3" />
}

export function ChartLegend() {
  return <Legend />
}

