(* What a DMA-fed matrix multiply ({!Dma_gemm}) should take, from the rates
   the suite measured on the subsystem, and the best of the variants for a
   problem.

   A step overlaps a block's work on the core (its tile multiplies or
   multiply-adds, and C's stores when the core writes C) with the chain
   carrying the next block in (and the previous block's C out, when the DMA
   writes C). While the core streams DTCM the DMA moves at the contended rate;
   what is left of the chain after the core is done moves at the full rate. *)

module G = Dma_gemm
module L = Coral_layout

type rates = {
  core_copy : float;  (** the core's RVV copy from SRAM to DTCM, B/cycle *)
  core_store : float;  (** the core's stores of C to SRAM, B/cycle *)
  dma : float;  (** the DMA alone, B/cycle *)
  dma_beside : float;  (** the DMA while the core streams DTCM *)
  dma_beside_copy : float;
      (** the DMA beside the core's copy, which writes DTCM half the time *)
  dma_start : float;  (** a chain's start, cycles *)
  descriptor : float;  (** each further descriptor of a chain, cycles *)
  repack_beat : float;
      (** the core's packing of row-major A in DTCM, elements a cycle when the
          stride is a beat (16 bytes) or more *)
  repack_peak : float;  (** the same with a short stride, elements a cycle *)
  step : float;  (** a step's own instructions: start, wait, ring halves *)
}

(* From dma.t (copy and overlap) and gemm_zvt_i8.t: the core's stores of C, a
   c-by-core block's cycles less its tile multiplies; the DMA beside the
   core's copy, a b-shared prologue's cycles; a step's own, the RVV blocks'
   cycles less their multiply-adds. The packing of row-major A, from
   gemm_row_major_a.t's cycles less the packed kernels': a strided store
   writes an element a beat, several when they share one. *)
let measured =
  {
    core_copy = 2.42;
    core_store = 2.34;
    dma = 16. /. 14.;
    dma_beside = 0.8;
    dma_beside_copy = 1.03;
    dma_start = 57.;
    descriptor = 8.;
    repack_beat = 0.9;
    repack_peak = 4.4;
    step = 50.;
  }

(* Each engine's operations a cycle with every operand in DTCM, from the peak
   suite's bursts. *)
let compute_rate : G.Engine.t -> float = function
  | Rvv_bf16 -> 12.6
  | Rvv_f32 -> 9.9
  | Rvv_i8 -> 31.3
  | Zvt_f32 -> 86.8
  | Zvt_i8 -> 347.2

(* [bytes] and [extra] cycles of descriptors by the DMA beside the core's
   [phases], each so many cycles at which the DMA moves so many bytes a
   cycle: the cycles until both are done. The chain's start overlaps the
   core's work too. *)
let beside ?(extra = 0.) r ~phases bytes =
  let busy = List.fold_left (fun n (cycles, _) -> n +. cycles) 0. phases in
  if bytes <= 0. then busy
  else
    (* in cycles of the DMA alone *)
    let need = (bytes /. r.dma) +. extra in
    let during =
      List.fold_left
        (fun n (cycles, rate) -> n +. (cycles *. rate /. r.dma))
        0. phases
    in
    busy +. Float.max 0. (r.dma_start +. need -. during)

let predict ?(r = measured) (t : G.t) =
  let f = float_of_int in
  let a, b = G.packed t in
  let resident =
    match t.geometry with
    | Resident_a _ -> f (String.length a)
    | Resident_b | Row_major_a -> f (String.length b)
    | Split_k _ -> f (G.chunk_bytes t)
  in
  let piece = f (G.piece t) and n = G.steps t in
  (* With K split, a step also brings a block's partial sums back and a share
     of the next chunk of B, each a further descriptor. *)
  let split =
    match t.geometry with
    | Split_k _ -> true
    | Resident_a _ | Resident_b | Row_major_a -> false
  in
  let c_in = if split then f (G.c_block t) else 0. in
  let share i =
    if split && (i / t.blocks) + 1 < G.chunks t then
      f (G.chunk_bytes t) /. f t.blocks
    else 0.
  in
  (* C's bytes a step by the core's stores, and by the DMA, a descriptor a
     row when A is resident. *)
  let c = f (G.c_step t) in
  let c_core = c *. f (G.Out.by_core ~rows:(G.rows t) t.out) /. f (G.rows t) in
  let c_dma = c -. c_core in
  let extra = f (max 0 (G.c_descriptors t - 1)) *. r.descriptor in
  let step_ops = f (G.ops t / n) in
  let prologue =
    match t.prologue with
    | Dma -> r.dma_start +. ((resident +. piece +. c_in) /. r.dma)
    | Shared ->
        let by_core =
          f (int_of_float resident * 3 / 4 / G.core_trip * G.core_trip)
        in
        beside r
          ~phases:[ (by_core /. r.core_copy, r.dma_beside_copy) ]
          (resident -. by_core +. piece +. c_in)
  in
  (* The step's multiplies stream DTCM, as does a row-major block's packing;
     its stores of C to SRAM do not. *)
  let repack =
    match t.geometry with
    | Row_major_a ->
        let e = f (Peak_kernels.Elem.bytes (G.shape t).src) in
        let stride = f (G.rows t) *. e in
        let rate =
          Float.min r.repack_peak (r.repack_beat *. Float.max 1. (16. /. stride))
        in
        [ (piece /. e /. rate, r.dma_beside) ]
    | Resident_a _ | Resident_b | Split_k _ -> []
  in
  let work =
    repack
    @ (step_ops /. compute_rate t.engine, r.dma_beside)
      :: (if c_core > 0. then [ (c_core /. r.core_store, r.dma) ] else [])
  in
  let steps =
    List.init n (fun i ->
        let p_in = if i + 1 < n then piece +. c_in else 0. in
        let c_out, extra = if i > 0 then (c_dma, extra) else (0., 0.) in
        let extra = if split then extra +. (2. *. r.descriptor) else extra in
        r.step +. beside ~extra r ~phases:work (p_in +. c_out +. share i))
  in
  let epilogue =
    if c_dma > 0. then r.dma_start +. (c_dma /. r.dma) +. extra else 0.
  in
  prologue +. List.fold_left ( +. ) 0. steps +. epilogue

(* The bytes of DTCM a variant needs, beside the runtime's own. *)
let dtcm t =
  List.fold_left
    (fun n (s : L.Slot.t) ->
      match s.home with Dtcm -> n + Int64.to_int s.size | Any | Ram -> n)
    0 (G.slots t)

(* Every implemented variant of a problem, M x W x K on an engine with A
   packed, that fits [budget] bytes of DTCM, fastest first by the model. *)
let variants ~budget ~engine ~m ~width ~depth =
  let rows = G.Engine.rows engine in
  let geometries =
    match G.Engine.lanes engine with
    | Some n ->
        (if width = n then [ G.Geometry.Resident_b ] else [])
        @
        if width mod n = 0 then [ G.Geometry.Resident_a { panel = n } ] else []
    | None ->
        G.Geometry.Resident_b
        :: List.filter_map
             (fun panel ->
               if panel <= width && width mod panel = 0 then
                 Some (G.Geometry.Resident_a { panel })
               else None)
             [ 32; 64; 128; 256 ]
  in
  (* Split K's partial sums go by the DMA. *)
  let outs : G.Geometry.t -> G.Out.t list = function
    | Resident_b when G.Engine.matrix engine ->
        G.Out.[ Core; Dma ]
        @ List.init ((rows / 4) - 1) (fun k -> G.Out.Split (4 * (k + 1)))
    | Resident_a _ | Resident_b | Row_major_a -> G.Out.[ Core; Dma ]
    | Split_k _ -> [ G.Out.Dma ]
  in
  let geometries =
    geometries
    @ List.filter_map
        (fun chunk ->
          if chunk < depth && depth mod chunk = 0 && m / rows >= 2 then
            Some (G.Geometry.Split_k { chunk })
          else None)
        [ 64; 128; 256; 512; 1024; 2048 ]
  in
  (* Split K keeps B's full width, as resident B does. *)
  let geometries =
    List.filter
      (function
        | G.Geometry.Split_k _ -> (
            match G.Engine.lanes engine with
            | Some n -> n = width
            | None -> true)
        | Resident_a _ | Resident_b | Row_major_a -> true)
      geometries
  in
  if m mod rows <> 0 then []
  else
    List.concat_map
      (fun geometry ->
        List.concat_map
          (fun prologue ->
            List.map
              (fun out ->
                {
                  G.blocks = m / rows;
                  depth;
                  engine;
                  geometry;
                  out;
                  prologue;
                  width;
                })
              (outs geometry))
          G.Prologue.[ Dma; Shared ])
      geometries
    |> List.filter (fun t -> dtcm t <= budget)
    |> List.map (fun t -> (predict t, t))
    |> List.sort (fun (x, _) (y, _) -> Float.compare x y)
