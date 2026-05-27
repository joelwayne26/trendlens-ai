import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all duration-300 ease-in-out disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default:
          "bg-white text-sky-700 border border-orange-200 shadow-sm hover:bg-orange-100 hover:border-orange-300 hover:text-orange-800 hover:shadow-md hover:-translate-y-0.5",
        destructive:
          "bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60 transition-all duration-300 ease-in-out hover:-translate-y-0.5",
        outline:
          "border border-sky-300 bg-white text-sky-700 shadow-xs hover:bg-sky-50 hover:text-sky-800 hover:border-sky-400 hover:shadow-md transition-all duration-300 ease-in-out hover:-translate-y-0.5",
        secondary:
          "bg-orange-100 text-orange-800 border border-orange-200 shadow-sm hover:bg-orange-200 hover:text-orange-900 hover:border-orange-300 hover:shadow-md transition-all duration-300 ease-in-out hover:-translate-y-0.5",
        ghost:
          "text-sky-700 hover:bg-sky-50 hover:text-sky-800 transition-all duration-300 ease-in-out",
        link: "text-sky-600 underline-offset-4 hover:underline hover:text-orange-600 transition-colors duration-300",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        sm: "h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-md px-6 has-[>svg]:px-4",
        icon: "size-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
