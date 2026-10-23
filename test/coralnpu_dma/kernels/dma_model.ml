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
  step : float;  (** a step's own instructions: start, wait, ring halves *)
}

(* From dma.t (copy and overlap) and gemm_zvt_i8.t: the core's stores of C, a
   c-by-core block's cycles less its tile multiplies; the DMA beside the
   core's copy, a b-shared prologue's cycles; a step's own, the RVV blocks'
   cycles less their multiply-adds. *)
let measured =
  {
    core_copy = 2.42;
    core_store = 2.34;
    dma = 16. /. 14.;
    dma_beside = 0.8;
    dma_beside_copy = 1.03;
    dma_start = 57.;
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

(* [bytes] by the DMA beside the core's [phases], each so many cycles at which
   the DMA moves so many bytes a cycle: the cycles until both are done. *)
let beside r ~phases bytes =
  let busy = List.fold_left (fun n (cycles, _) -> n +. cycles) 0. phases in
  if bytes <= 0. then busy
  else
    let during =
      List.fold_left (fun n (cycles, rate) -> n +. (cycles *. rate)) 0. phases
    in
    if bytes <= during then busy +. r.dma_start
    else busy +. r.dma_start +. ((bytes -. during) /. r.dma)

let b_bytes t = String.length (snd (G.packed t))

let predict ?(r = measured) (t : G.t) =
  let f = float_of_int in
  let a = f (G.a_block t) and c = f (G.c_block t) and b = f (b_bytes t) in
  let block_ops = f (G.ops t / t.blocks) in
  let prologue =
    match t.prologue with
    | Dma -> r.dma_start +. ((b +. a) /. r.dma)
    | Shared ->
        let by_core = f (b_bytes t * 3 / 4 / G.core_trip * G.core_trip) in
        beside r
          ~phases:[ (by_core /. r.core_copy, r.dma_beside_copy) ]
          (b -. by_core +. a)
  in
  (* The block's multiplies stream DTCM; its stores of C to SRAM do not. *)
  let work =
    (block_ops /. compute_rate t.engine, r.dma_beside)
    :: (match t.out with Core -> [ (c /. r.core_store, r.dma) ] | Dma -> [])
  in
  let steps =
    List.init t.blocks (fun i ->
        let a_in = if i + 1 < t.blocks then a else 0. in
        let c_out = match t.out with Dma when i > 0 -> c | _ -> 0. in
        r.step +. beside r ~phases:work (a_in +. c_out))
  in
  let epilogue =
    match t.out with Dma -> r.dma_start +. (c /. r.dma) | Core -> 0.
  in
  prologue +. List.fold_left ( +. ) 0. steps +. epilogue

(* The bytes of DTCM a variant needs, beside the runtime's own. *)
let dtcm t =
  List.fold_left
    (fun n (s : L.Slot.t) ->
      match s.home with Dtcm -> n + Int64.to_int s.size | Any | Ram -> n)
    0 (G.slots t)

(* Every implemented variant of a problem, M x W x K on an engine, that fits
   [budget] bytes of DTCM, fastest first by the model. *)
let variants ~budget ~engine ~m ~width ~depth =
  let rows = G.Engine.rows engine in
  if m mod rows <> 0 then []
  else
    List.concat_map
      (fun prologue ->
        List.map
          (fun out ->
            { G.blocks = m / rows; depth; engine; out; prologue; width })
          G.Out.[ Core; Dma ])
      G.Prologue.[ Dma; Shared ]
    |> List.filter (fun t -> dtcm t <= budget)
    |> List.map (fun t -> (predict t, t))
    |> List.sort (fun (x, _) (y, _) -> Float.compare x y)
