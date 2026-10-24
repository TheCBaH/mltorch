(* Raw transfer speed between the subsystem's memories: the DMA engine against
   the core's own loads and stores, each case timed inside the kernel.

   A sweep is one kernel run over a list of cases on one path (source and
   destination region). The table holds, in order:

   - 0 [cases]: the case count, then one word per case: a descriptor chain's
     address (DMA) or a byte count (the core's kernels);
   - 1 [chains]: the DMA's descriptors, built by the host for the layout;
   - 2 [src], 3 [dst]: the buffers, each as large as the largest case;
   - 4 [timings]: two words per case, the cycles from just before the transfer
     starts to just after it is seen complete, and the DMA's final [STATUS]
     (0 for the core's).

   The largest case comes last and fills [dst] from [src], so [dst]'s hash
   checks the transfer that matters most; the read-only and write-only kernels
   leave [dst] zero. Only caller-saved registers are written. *)

open Peak_kernels
module D = Coral_dma

(* The memory a buffer sits in, and the two ends of a transfer. Alphabetical. *)
module Memory = struct
  type t = Dtcm | Sram

  let home : t -> L.Home.t = function Dtcm -> Dtcm | Sram -> Ram
  let name = function Dtcm -> "dtcm" | Sram -> "sram"
end

module Path = struct
  type t = { dst : Memory.t; src : Memory.t }

  let name p = Printf.sprintf "%s->%s" (Memory.name p.src) (Memory.name p.dst)
end

(* What moves the bytes. Alphabetical. *)
module Engine = struct
  type t =
    | Dma of D.Beat.t
    | Rvv_copy of int  (** LMUL: [vle8.v] then [vse8.v], two groups a trip *)
    | Rvv_read of int  (** [vle8.v] only *)
    | Rvv_write of int  (** [vse8.v] only, of zeros *)
    | Scalar_copy  (** [lw] then [sw], four words a trip *)

  let name = function
    | Dma b -> Printf.sprintf "dma/%d" (D.Beat.bytes b)
    | Rvv_copy m -> Printf.sprintf "rvv-copy/m%d" m
    | Rvv_read m -> Printf.sprintf "rvv-read/m%d" m
    | Rvv_write m -> Printf.sprintf "rvv-write/m%d" m
    | Scalar_copy -> "scalar-copy"

  (* Bytes one trip of a core loop moves; a case is a whole number of trips. *)
  let trip = function
    | Dma b -> D.Beat.bytes b
    | Rvv_copy m | Rvv_read m | Rvv_write m -> 2 * 16 * m
    | Scalar_copy -> 16

  let copies = function
    | Dma _ | Rvv_copy _ | Scalar_copy -> true
    | Rvv_read _ | Rvv_write _ -> false
end

(* A case: one chain of descriptors (DMA), each [piece] bytes, [length] in all;
   the core's kernels take [length] as one transfer. *)
module Case = struct
  type t = { length : int; piece : int }

  let whole length = { length; piece = length }
end

type sweep = { cases : Case.t list; engine : Engine.t; path : Path.t }

let cases_slot = 0
let chains_slot = 1
let src_slot = 2
let dst_slot = 3
let timings_slot = 4
let largest s = List.fold_left (fun m (c : Case.t) -> max m c.length) 0 s.cases
let chain_count (c : Case.t) = c.length / c.piece

let slots s =
  let size = largest s in
  let chains =
    match s.engine with
    | Dma _ ->
        D.Descriptor.size
        * List.fold_left (fun n c -> n + chain_count c) 0 s.cases
    | Rvv_copy _ | Rvv_read _ | Rvv_write _ | Scalar_copy -> 16
  in
  let n = List.length s.cases in
  L.Placement.
    [
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int (4 * (n + 1)));
      L.Slot.make ~home:Ram ~align:32L In (Int64.of_int chains);
      L.Slot.make ~home:(Memory.home s.path.src) ~align:64L In
        (Int64.of_int size);
      L.Slot.make ~home:(Memory.home s.path.dst) ~align:64L Out
        (Int64.of_int size);
      L.Slot.make ~home:Dtcm ~align:16L Out (Int64.of_int (8 * n));
    ]

(* The source's bytes: a different word everywhere, so an offset or order
   error changes the hash. *)
let source s =
  let n = largest s in
  String.init n (fun i ->
      Char.chr (((i * 131) + ((i lsr 8) * 7) + (i lsr 16)) land 0xFF))

let expected s =
  if Engine.copies s.engine then source s else String.make (largest s) '\000'

let words l =
  let b = Bytes.create (4 * List.length l) in
  List.iteri (fun k w -> Bytes.set_int32_le b (4 * k) (Int32.of_int w)) l;
  Bytes.to_string b

(* The input slots' bytes for a layout: the case table and, for the DMA, the
   chains, laid one after another in their slot. *)
let inputs p s (l : L.t) =
  let address k = (List.nth l.slots k).L.Placed_slot.address in
  let src = address src_slot and dst = address dst_slot in
  match s.engine with
  | Dma beat ->
      let at = address chains_slot in
      let open Err.Syntax in
      let* chains, _ =
        Err.List.fold_left
          (fun (acc, at) (c : Case.t) ->
            let descriptors =
              List.init (chain_count c) (fun k ->
                  let off = Int64.of_int (k * c.piece) in
                  D.Descriptor.copy ~beat ~src:(Int64.add src off)
                    ~dst:(Int64.add dst off) (Int64.of_int c.piece))
            in
            let* bytes = D.chain p ~at descriptors in
            Err.return
              ( (at, bytes) :: acc,
                Int64.add at (Int64.of_int (String.length bytes)) ))
          ([], at) s.cases
      in
      let chains = List.rev chains in
      Err.return
        [
          words
            (List.length s.cases
            :: List.map (fun (a, _) -> Int64.to_int a) chains);
          String.concat "" (List.map snd chains);
          source s;
        ]
  | Rvv_copy _ | Rvv_read _ | Rvv_write _ | Scalar_copy ->
      Err.return
        [
          words
            (List.length s.cases
            :: List.map (fun (c : Case.t) -> c.length) s.cases);
          "";
          source s;
        ]

(* {1 The kernels} *)

let andi dst src n = Item.insn (Insn.Andi { dst; src; imm = imm12 n })
let beqz src target = Item.insn (Insn.Beqz { src; target })
let fence = Item.insn Insn.Fence
let mcycle dst = Item.insn (Insn.Csrr { dst; csr = Mcycle })
let sub dst lhs rhs = Item.insn (Insn.Sub { dst; lhs; rhs })

(* [a1] the case cursor, [a2] the timings cursor, [a3] the cases left, [a6]
   and [a7] the buffers; then [body] per case, between two reads of [mcycle]
   into [t3] and [t4], with [t5] the case's word; then the timing and [t2],
   the status, stored. *)
let sweep_items ~setup ~body =
  let case = Label.named "case" in
  let open Err.Syntax in
  let* setup = setup in
  Ok
    (start_items
    @ [
        lw Gpr.a1 Gpr.a0 (4 * cases_slot);
        lw Gpr.a2 Gpr.a0 (4 * timings_slot);
        lw Gpr.a6 Gpr.a0 (4 * src_slot);
        lw Gpr.a7 Gpr.a0 (4 * dst_slot);
        lw Gpr.a3 Gpr.a1 0;
        addi Gpr.a1 Gpr.a1 4;
      ]
    @ setup
    @ [ Item.label case; lw Gpr.t5 Gpr.a1 0; li Gpr.t2 0; fence; mcycle Gpr.t3 ]
    @ body_items body
    @ [
        fence;
        mcycle Gpr.t4;
        sub Gpr.t4 Gpr.t4 Gpr.t3;
        sw Gpr.t4 Gpr.a2 0;
        sw Gpr.t2 Gpr.a2 4;
        addi Gpr.a2 Gpr.a2 8;
        addi Gpr.a1 Gpr.a1 4;
        addi Gpr.a3 Gpr.a3 (-1);
        bnez Gpr.a3 case;
        li Gpr.a0 0;
        Item.insn Ret;
      ])

(* The chain at [t5] started, then [STATUS] polled until done, into [t2]. *)
let dma_items () =
  let wait = Label.named "wait" in
  sweep_items
    ~setup:
      (Result.map
         (fun base -> List.map Item.insn base @ [ li Gpr.a5 D.ctrl_start ])
         (Insn.load_imm Gpr.a4 (Int64.to_int D.base)))
    ~body:
      [
        sw Gpr.t5 Gpr.a4 (D.Register.offset Desc_addr);
        sw Gpr.a5 Gpr.a4 (D.Register.offset Ctrl);
        Item.label wait;
        lw Gpr.t2 Gpr.a4 (D.Register.offset Status);
        andi Gpr.t0 Gpr.t2 D.status_done;
        beqz Gpr.t0 wait;
      ]

let log2 n =
  let rec go k = if 1 lsl k >= n then k else go (k + 1) in
  go 0

(* [t6] trips of [trip] bytes from [t1] (the source cursor) to [a5] (the
   destination's; [t2] holds the status word). *)
let core_items engine =
  let loop = Label.named "trip" in
  let trip = Engine.trip engine in
  let trips =
    [ srli Gpr.t6 Gpr.t5 (log2 trip); mv Gpr.t1 Gpr.a6; mv Gpr.a5 Gpr.a7 ]
  in
  let vgroup m =
    let step = 16 * m in
    let e8 = Vtype.Sew.E8 in
    ( [ vset_item 8 m ],
      step,
      fun ~load ~store ->
        (if load then
           [
             vle e8 Vreg.v0 Gpr.t1;
             addi Gpr.t1 Gpr.t1 step;
             vle e8 Vreg.v8 Gpr.t1;
             addi Gpr.t1 Gpr.t1 step;
           ]
         else [])
        @
        if store then
          [
            vse e8 Vreg.v0 Gpr.a5;
            addi Gpr.a5 Gpr.a5 step;
            vse e8 Vreg.v8 Gpr.a5;
            addi Gpr.a5 Gpr.a5 step;
          ]
        else [] )
  in
  let tail = [ addi Gpr.t6 Gpr.t6 (-1); bnez Gpr.t6 loop ] in
  let setup, work =
    match (engine : Engine.t) with
    | Dma _ -> invalid_arg "Dma_copy.core_items: a DMA engine"
    | Rvv_copy m ->
        let set, _, w = vgroup m in
        (set, w ~load:true ~store:true)
    | Rvv_read m ->
        let set, _, w = vgroup m in
        (set, w ~load:true ~store:false)
    | Rvv_write m ->
        let set, _, w = vgroup m in
        (set @ [ vmv_v_i Vreg.v0; vmv_v_i Vreg.v8 ], w ~load:false ~store:true)
    | Scalar_copy ->
        (* [t5]'s byte count is in [t6] by now, so it carries a word too. *)
        let data = Gpr.[ a4; t0; t4; t5 ] in
        ( [],
          List.mapi (fun k r -> lw r Gpr.t1 (4 * k)) data
          @ List.mapi (fun k r -> sw r Gpr.a5 (4 * k)) data
          @ [ addi Gpr.t1 Gpr.t1 16; addi Gpr.a5 Gpr.a5 16 ] )
  in
  sweep_items ~setup:(Ok setup) ~body:(trips @ [ Item.label loop ] @ work @ tail)

let items s () =
  match s.engine with
  | Dma _ -> dma_items ()
  | Rvv_copy _ | Rvv_read _ | Rvv_write _ | Scalar_copy -> core_items s.engine

let name s = Printf.sprintf "%s %s" (Path.name s.path) (Engine.name s.engine)
