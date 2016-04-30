#I "packages\\Alea.CUDA\\lib\\net40"
#r "System.Configuration.dll"
#r "Alea.CUDA.dll"

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open System.Threading.Tasks

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- __SOURCE_DIRECTORY__ + "\packages\Alea.CUDA\private"
Alea.CUDA.Settings.Instance.Resource.Path <- __SOURCE_DIRECTORY__ + @"\release"

let generate ln =
    let n = ln*ln
    let rng = Random()
    let x = Array.init n (fun l -> if (l%(ln+1)) = 0 then 0.f else float32 <| rng.NextDouble())
    x

let floydwarshall (l:int, mat:float32[]) =
   let a = Array.copy mat
   let p = Array.create mat.Length (-1)
   for k in 0 .. (l-1) do
      for i in 0 .. (l-1) do
         for j in 0 .. (l-1) do
            if (a.[i*l+k] + a.[k*l+j]) < a.[i*l+j] then a.[i*l+j] <- (a.[i*l+k] + a.[k*l+j])
                                                        p.[i*l+j] <- k
   a,p

let parallelFW (l:int, mat : float32[]) = 
   let a = Array.copy mat
   let p = Array.create (l*l) (-1)
   for k in 0 .. (l-1) do
     Parallel.For(0, (l-1), (fun i ->
          for j in 0 .. (l-1) do
             if (a.[i*l+k] + a.[k*l+j]) < a.[i*l+j] then a.[i*l+j] <- (a.[i*l+k] + a.[k*l+j])
                                                         p.[i*l+j] <- k)) 
     |> ignore
   a,p

type FWModule(target:GPUModuleTarget, tileDim:int) = //, blockRows:int) =
    inherit GPUModule(target)
    
    [<Kernel;ReflectedDefinition>]
    member this.FloydWKernel (size:int) (k:int) (paths:deviceptr<int>) (graph:deviceptr<float32>) =
        let col = blockIdx.x * tileDim + threadIdx.x //col is index i, blockIdx.y is index j
        let row = blockIdx.y
        
        if col >= size then ()

        let index  = size * row + col
        let best = __shared__.Variable<float32>()

        if threadIdx.x = 0 then best := graph.[size*row+k]
        __syncthreads()
        
        let tmp = graph.[k*size+col]
        
        if tmp = infinityf then ()
        let cur = !best + tmp
        if cur < graph.[index] then graph.[index] <- cur
                                    paths.[index] <- k

    member this.LaunchParams size =
        let blockdim = dim3(tileDim)
        let griddim = dim3(divup size tileDim, size)
        LaunchParam(griddim, blockdim)        

    member this.FW2(size:int, graph:float32[])= 
         use deviceGraph = this.GPUWorker.Malloc(graph)
         let paths = Array.create (graph.Length) (-1)
         use devicePaths = this.GPUWorker.Malloc(paths)
         let lp = this.LaunchParams size
         for k in 0 .. size-1 do
            this.GPULaunch <@ this.FloydWKernel @> lp size k devicePaths.Ptr deviceGraph.Ptr
         deviceGraph.Gather(),devicePaths.Gather()

    [<Kernel;ReflectedDefinition>]
    member this.FWKernel (size:int) (k:int) (graph:deviceptr<float32>) (paths:deviceptr<int>)=
        let i = blockIdx.y * blockDim.y + threadIdx.y
        let j = blockIdx.x * blockDim.x + threadIdx.x
        if i >= size || j >= size then ()
        let kj = k*size + j
        let ij = i*size + j
        let ik = i*size + k
        if graph.[ik] + graph.[kj] < graph.[ij] then
           graph.[ij] <- graph.[ik] + graph.[kj]
           paths.[ij] <- k

    member this.LaunchParams2 size =
        let nthreads = tileDim |> float |> sqrt |> int
        let nblocks = divup size nthreads
        let blockdim = dim3(nthreads,nthreads)
        let griddim = dim3(nblocks,nblocks)
        LaunchParam(griddim,blockdim)

    member this.FW(size:int, graph:float32[]) =
        use deviceGraph = this.GPUWorker.Malloc(graph)
        let paths = Array.create (graph.Length) (-1)
        use devicePaths = this.GPUWorker.Malloc(paths)
        let lp = this.LaunchParams size
        for k in 0 .. size-1 do
           this.GPULaunch <@ this.FWKernel @> lp size k deviceGraph.Ptr devicePaths.Ptr
        deviceGraph.Gather(),devicePaths.Gather()

let tileDim = 256

let apsp = new FWModule(GPUModuleTarget.DefaultWorker, tileDim)

(*let MatrixTest () = 
    let validate (dimA:int*int) =
        let wA, hA = dimA
        let sizeA = wA * hA
        let rng = Random()
        let m (l,c) = Array2D.init<float> l c (fun x y -> if x=y then 0.0 else rng.NextDouble())
        let dAB = matMult.Mult (m dimA)
        //let err = Array.map2 (fun d h -> abs (d - h)) dAB hAB |> Array.max 
        printfn "%A \n %A" dAB m

    let dimensions = [(512,512)]//; (512, 512); (1024, 1024); (2048, 2048)]
    List.iter validate dimensions*)

let arr (z:float32[]) (y:float32[]) =
   let l = z.Length
   let v = Array.zeroCreate l
   for i in 0..(l-1) do
      v.[i] <- z.[i] = y.[i]
   v

let x1,x2,x3 = (fun x y z -> generate x, generate y, generate z)  1024 2048 4096
