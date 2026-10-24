(* The DMA in flight while the core works: does either slow the other?

   Each case optionally starts one chain (SRAM to DTCM, [dst]), then runs
   [trips] of one workload, then waits for the chain. It records the cycles to
   the workload's end and to the chain's completion seen, from a common start,
   and the final [STATUS]. A case with no chain times the workload alone; one
   with no trips times the chain alone. The table holds:

   - 0 [cases]: the count, then per case the chain's address (0 for none) and
     the trips;
   - 1 [chains]; 2 [src] (SRAM); 3 [dst] (DTCM);
   - 4 [work]: the 256-byte buffer the memory workloads use, in DTCM or SRAM;
   - 5 [timings]: three words per case. *)

open Peak_kernels
module D = Coral_dma

(* What the core does meanwhile. Alphabetical. *)
module Work = struct
  type t =
    | Alu  (** four [addi]s a trip: no memory at all *)
    | Dtcm_read  (** two [vle8.v] at m8 from DTCM, 256 bytes a trip *)
    | Dtcm_write  (** two [vse8.v] at m8 to DTCM *)
    | Sram_read  (** two [vle8.v] at m8 from SRAM *)

  let name = function
    | Alu -> "alu"
    | Dtcm_read -> "dtcm-read"
    | Dtcm_write -> "dtcm-write"
    | Sram_read -> "sram-read"

  let home : t -> L.Home.t = function
    | Alu | Dtcm_read | Dtcm_write -> Dtcm
    | Sram_read -> Ram
end

module Case = struct
  type t = { dma : bool; trips : int }
end

type sweep = { cases : Case.t list; length : int; work : Work.t }

let timings_slot = 5
let dst_slot = 3

let slots s =
  let n = List.length s.cases in
  let chains = List.length (List.filter (fun (c : Case.t) -> c.dma) s.cases) in
  L.Placement.
    [
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int (4 * ((2 * n) + 1)));
      L.Slot.make ~home:Ram ~align:32L In
        (Int64.of_int (max 16 (D.Descriptor.size * chains)));
      L.Slot.make ~home:Ram ~align:64L In (Int64.of_int s.length);
      L.Slot.make ~home:Dtcm ~align:64L Out (Int64.of_int s.length);
      L.Slot.make ~home:(Work.home s.work) ~align:64L In 256L;
      L.Slot.make ~home:Dtcm ~align:16L Out (Int64.of_int (12 * n));
    ]

let source s =
  Dma_copy.source
    {
      Dma_copy.cases = [ Dma_copy.Case.whole s.length ];
      engine = Scalar_copy;
      path = { src = Sram; dst = Dtcm };
    }

(* Every chain copies the whole source, so [dst] ends up equal to it if any
   case has one. *)
let expected s =
  if List.exists (fun (c : Case.t) -> c.dma) s.cases then source s
  else String.make s.length '\000'

let inputs p s (l : L.t) =
  let address k = (List.nth l.slots k).L.Placed_slot.address in
  let src = address 2 and dst = address dst_slot and at = address 1 in
  let open Err.Syntax in
  let* chains, table, _ =
    Err.List.fold_left
      (fun (chains, table, at) (c : Case.t) ->
        if c.dma then
          let* bytes =
            D.chain p ~at
              [ D.Descriptor.copy ~src ~dst (Int64.of_int s.length) ]
          in
          Err.return
            ( bytes :: chains,
              c.trips :: Int64.to_int at :: table,
              Int64.add at (Int64.of_int (String.length bytes)) )
        else Err.return (chains, c.trips :: 0 :: table, at))
      ([], [], at) s.cases
  in
  Err.return
    [
      Dma_copy.words (List.length s.cases :: List.rev table);
      String.concat "" (List.rev chains);
      source s;
      "";
      String.make 256 '\001';
    ]

let andi = Dma_copy.andi
let beqz = Dma_copy.beqz
let fence = Dma_copy.fence
let mcycle = Dma_copy.mcycle
let sub = Dma_copy.sub

(* [a1] the cases, [a2] the timings, [a3] the cases left, [a4] the DMA's
   registers, [a5] its start word, [a6] the work buffer; per case [t5] the
   chain, [t6] the trips, [t3] the start. *)
let items s () =
  let open Err.Syntax in
  let case = Label.named "case"
  and no_dma = Label.named "no_dma"
  and trip = Label.named "trip"
  and no_work = Label.named "no_work"
  and wait = Label.named "wait"
  and done_ = Label.named "done" in
  let e8 = Vtype.Sew.E8 in
  let work =
    match s.work with
    | Alu ->
        [
          addi Gpr.t0 Gpr.t0 1;
          addi Gpr.t1 Gpr.t1 1;
          addi Gpr.t4 Gpr.t4 1;
          addi Gpr.a7 Gpr.a7 1;
        ]
    | Dtcm_read | Sram_read -> [ vle e8 Vreg.v0 Gpr.a6; vle e8 Vreg.v8 Gpr.t1 ]
    | Dtcm_write -> [ vse e8 Vreg.v16 Gpr.a6; vse e8 Vreg.v24 Gpr.t1 ]
  in
  let* base = Insn.load_imm Gpr.a4 (Int64.to_int D.base) in
  Ok
    (start_items
    @ [
        lw Gpr.a1 Gpr.a0 0;
        lw Gpr.a2 Gpr.a0 (4 * timings_slot);
        lw Gpr.a6 Gpr.a0 16;
        addi Gpr.t1 Gpr.a6 128;
        lw Gpr.a3 Gpr.a1 0;
        addi Gpr.a1 Gpr.a1 4;
      ]
    @ List.map Item.insn base
    @ [
        li Gpr.a5 D.ctrl_start;
        vset_item 8 8;
        vmv_v_i Vreg.v16;
        vmv_v_i Vreg.v24;
        Item.label case;
        lw Gpr.t5 Gpr.a1 0;
        lw Gpr.t6 Gpr.a1 4;
        li Gpr.t2 0;
        fence;
        mcycle Gpr.t3;
        beqz Gpr.t5 no_dma;
        sw Gpr.t5 Gpr.a4 (D.Register.offset Desc_addr);
        sw Gpr.a5 Gpr.a4 (D.Register.offset Ctrl);
        Item.label no_dma;
        beqz Gpr.t6 no_work;
      ]
    @ body_items
        ([ Item.label trip ]
        @ work
        @ [ addi Gpr.t6 Gpr.t6 (-1); bnez Gpr.t6 trip ])
    @ [
        Item.label no_work;
        fence;
        mcycle Gpr.t4;
        sub Gpr.t4 Gpr.t4 Gpr.t3;
        sw Gpr.t4 Gpr.a2 0;
        beqz Gpr.t5 done_;
        Item.label wait;
        lw Gpr.t2 Gpr.a4 (D.Register.offset Status);
        andi Gpr.t0 Gpr.t2 D.status_done;
        beqz Gpr.t0 wait;
        Item.label done_;
        mcycle Gpr.t4;
        sub Gpr.t4 Gpr.t4 Gpr.t3;
        sw Gpr.t4 Gpr.a2 4;
        sw Gpr.t2 Gpr.a2 8;
        addi Gpr.a2 Gpr.a2 12;
        addi Gpr.a1 Gpr.a1 8;
        addi Gpr.a3 Gpr.a3 (-1);
        bnez Gpr.a3 case;
        li Gpr.a0 0;
        Item.insn Ret;
      ])

let name s = Printf.sprintf "overlap %s %d" (Work.name s.work) s.length
