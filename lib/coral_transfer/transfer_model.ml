module C = Contraction
module P = Transfer_plan

module Rates = struct
  type t = {
    vector_fused : float;
    vector_ordered : float;
    vector_row_fused : float;
    vector_row_ordered : float;
    step : float;
    a_dtcm : float;
    a_sram : float;
    a_bf16 : float;
    b_beat_dtcm : float;
    b_row_sram : float;
    b_beat_sram : float;
    b_element_dtcm : float;
    b_element_sram : float;
    group : float;
    c_beat : float;
    dma : float;
    dma_beside : float;
    dma_start : float;
    contention : float;
    core_copy : float;
    dma_beside_copy : float;
    c_beat_dtcm : float;
  }

  (* Fitted by Nelder-Mead on relative squared error to the 85 plans of the
     16 shapes in the transfer grid (test/machine_rivet_riscv32_coral_transfer,
     subsystem-vector-normal, v0.0.5), the DMA's own rate and start held at
     the DMA suite's measurements; the bfloat16 widening is its two
     instructions, unmeasured. Worst error 8.8% in the fit; left out a shape at
     a time, 6.1% at the median shape and 19.1% at the worst, an ordered one:
     the model ranks plans better than it times them. Fitted before the k
     loop was pipelined; against the pipelined function the same rates are
     9.5% off at worst and still choose the best plan of every shape, and a
     refit does no better (11.6%), so they stand. The fit put a
     contiguous row's own SRAM access and a group's own instructions at zero:
     the step and C's stores carry them. *)
  let fitted =
    {
      vector_fused = 0.177;
      vector_ordered = 0.235;
      vector_row_fused = 1.671;
      vector_row_ordered = 5.708;
      step = 14.103;
      a_dtcm = 1.814;
      a_sram = 5.578;
      a_bf16 = 2.;
      b_beat_dtcm = 0.318;
      b_row_sram = 0.;
      b_beat_sram = 4.182;
      b_element_dtcm = 0.843;
      b_element_sram = 4.816;
      group = 0.;
      c_beat = 12.464;
      dma = 16. /. 14.;
      dma_beside = 0.8;
      dma_start = 57.;
      contention = 0.216;
      (* the DMA suite's measured rates for the shared copy; C's stores to
         DTCM a beat a cycle, as B's loads from it, unmeasured *)
      core_copy = 2.42;
      dma_beside_copy = 1.03;
      c_beat_dtcm = 1.;
    }
end

let f = float_of_int
let cdiv a b = (a + b - 1) / b

(* One k step of a group of [rows] rows over a tile of [lanes]: the longer
   of its vector work and its loads, which overlap, then its own scalar
   instructions. *)
let k_step (r : Rates.t) (d : C.t) (p : P.t) ~a_sram ~rows ~lanes =
  let size = Int64.to_int (C.Element.bytes d.C.format) in
  let lane_beats = f (cdiv lanes 4) in
  let vector =
    f rows
    *.
    match d.C.numerics with
    | C.Numerics.Fused_sequential ->
        r.Rates.vector_row_fused +. (lane_beats *. r.Rates.vector_fused)
    | C.Numerics.Ordered ->
        r.Rates.vector_row_ordered +. (lane_beats *. r.Rates.vector_ordered)
  in
  let a =
    f rows
    *. ((if a_sram then r.Rates.a_sram else r.Rates.a_dtcm)
       +. if d.C.format = C.Element.Bf16 then r.Rates.a_bf16 else 0.)
  in
  let contiguous = Int64.equal d.C.b.C.Operand.outer 1L in
  let beats = f (cdiv (lanes * size) 16) in
  let b =
    match (p.P.b, contiguous) with
    | (P.B.Copied | P.B.Shared), true -> beats *. r.Rates.b_beat_dtcm
    | P.B.In_place, true -> r.Rates.b_row_sram +. (beats *. r.Rates.b_beat_sram)
    | (P.B.Copied | P.B.Shared), false -> f lanes *. r.Rates.b_element_dtcm
    | P.B.In_place, false -> f lanes *. r.Rates.b_element_sram
  in
  Float.max vector (a +. b) +. r.Rates.step

(* [rows] rows of one batch: every tile, every group, every k step, and each
   group's own instructions and C's stores *)
let product r (d : C.t) p ~a_sram rows =
  let n = Int64.to_int d.C.n and k = f (Int64.to_int d.C.k) in
  let g = P.group_rows d in
  List.fold_left ( +. ) 0.
    (List.init (cdiv n 32) (fun t ->
         let lanes = min 32 (n - (32 * t)) in
         let groups =
           List.init (cdiv rows g) (fun j -> min g (rows - (j * g)))
         in
         List.fold_left
           (fun acc gr ->
             acc
             +. (k *. k_step r d p ~a_sram ~rows:gr ~lanes)
             +. r.Rates.group
             +. f (gr * cdiv (lanes * 4) 16)
                *.
                match p.P.out with
                | P.Out.Core -> r.Rates.c_beat
                | P.Out.Dma -> r.Rates.c_beat_dtcm)
           0. groups))

let predict ?(rates = Rates.fitted) (d : C.t) (p : P.t) =
  let r = rates in
  let m = Int64.to_int d.C.m and batch = f (Int64.to_int d.C.batch) in
  let copy bytes = r.Rates.dma_start +. (f bytes /. r.Rates.dma) in
  let copies =
    (match p.P.a with P.A.Copied -> copy (P.span d `A) | _ -> 0.)
    +.
    match p.P.b with
    | P.B.Copied -> copy (P.span d `B)
    | P.B.In_place -> 0.
    | P.B.Shared ->
        (* the core copies three quarters, by 256-byte trips, beside the
           DMA's rest *)
        let span = P.span d `B in
        let by_core = span * 3 / 4 / 256 * 256 in
        Float.max
          (f by_core /. r.Rates.core_copy)
          (r.Rates.dma_start +. (f (span - by_core) /. r.Rates.dma_beside_copy))
  in
  let per_batch =
    match p.P.a with
    | P.A.In_place -> product r d p ~a_sram:true m
    | P.A.Copied -> product r d p ~a_sram:false m
    | P.A.Streamed { rows } ->
        let blocks = cdiv m rows in
        let last = m - ((blocks - 1) * rows) in
        let bytes = rows * P.row_bytes d in
        let full = product r d p ~a_sram:false rows in
        (* each later block's chain beside the block before: A's next block
           and, with C by the DMA, the block before's C; it costs what the
           product does not hide and, when the product reads B from SRAM, a
           contention per byte. A last chain carries the last blocks' C. *)
        let c_block = rows * P.c_row_bytes d in
        let chain, c_after =
          match p.P.out with
          | P.Out.Core -> (bytes, 0.)
          | P.Out.Dma ->
              ( bytes + c_block,
                r.Rates.dma_start
                +. f (((min blocks 2 - 1) * c_block) + (last * P.c_row_bytes d))
                   /. r.Rates.dma )
        in
        let beside =
          Float.max 0. ((f chain /. r.Rates.dma_beside) -. full)
          +.
          match p.P.b with
          | P.B.In_place -> f chain *. r.Rates.contention
          | P.B.Copied | P.B.Shared -> 0.
        in
        copy (if blocks = 1 then last * P.row_bytes d else bytes)
        +. (f (blocks - 1) *. (full +. beside))
        +. product r d p ~a_sram:false last
        +. c_after
  in
  copies +. (batch *. per_batch)

let choose ?rates ~budget d =
  List.fold_left
    (fun (best, cost) p ->
      let c = predict ?rates d p in
      if c < cost then (p, c) else (best, cost))
    (P.in_place, predict ?rates d P.in_place)
    (P.candidates ~budget d)
  |> fst
