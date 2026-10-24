(* The DMA's descriptor polling ({!Coral_dma.Poll}): can a chain wait on the
   core, and what does the waiting cost either side?

   Each case starts its chain (if any), runs [work] trips of the core's DTCM
   read stream, reads [CUR_DESC], spins [delay] trips, then writes 1 to its
   flag (if any) and waits: for [STATUS.done], or for a token word the chain's
   last descriptor writes into DTCM. It records the stream's cycles, the
   cycles from the flag's write to the end of the wait, [CUR_DESC] after the
   stream, and the final [STATUS]. The table holds:

   - 0 [cases]: the count, then five words a case: the chain (0 for none), the
     flag's address (0 for none), [delay], [work], the token's address (0 to
     wait on [STATUS]);
   - 1 [chains]; 2 [src] (SRAM); 3 [dst] (DTCM);
   - 4 [work]: the 256-byte buffer the stream reads (DTCM);
   - 5 [timings]: four words a case;
   - 6 SRAM flags, 7 DTCM flags, 8 the token's source (SRAM), 9 the tokens
     (DTCM): a word each per case, zero until written. *)

open Peak_kernels
module D = Coral_dma

(* What a case's chain does. Alphabetical. *)
module Chain = struct
  type t =
    | Copy  (** [length] bytes SRAM to DTCM *)
    | Gated  (** a one-beat descriptor polling the case's flag, then the copy *)
    | Handshake  (** a gate on the SRAM flag, then a 16-byte token to DTCM *)
    | No_chain
    | Polled_copy of [ `Dtcm | `Sram ]
        (** the copy, every beat polling a word whose test always passes *)
    | Token  (** the 16-byte token alone, as an unconditioned chain *)
end

module Flag = struct
  type t = Dtcm | No_flag | Sram
end

module Case = struct
  type t = {
    chain : Chain.t;
    delay : int;
    flag : Flag.t;
    name : string;
    token : bool;  (** wait for the token in DTCM, not [STATUS] *)
    work : int;
  }
end

let length = 4096
let timings_slot = 5
let dst_slot = 3

let cases =
  let c ?(delay = 0) ?(flag = Flag.No_flag) ?(token = false) ?(work = 0) name
      chain =
    { Case.chain; delay; flag; name; token; work }
  in
  let gated = Chain.Gated in
  [
    c "copy alone" Copy;
    c "gated copy, released after a delay" gated ~flag:Sram ~delay:500;
    c "copy, every beat polling SRAM" (Polled_copy `Sram);
    c "copy, every beat polling DTCM" (Polled_copy `Dtcm);
    c "core stream alone" No_chain ~work:64;
    c "core stream, the DMA polling an SRAM flag" gated ~flag:Sram ~work:64;
    c "core stream, the DMA polling a DTCM flag" gated ~flag:Dtcm ~work:64;
    c "handshake: flag in SRAM, token back in DTCM" Handshake ~flag:Sram
      ~token:true ~delay:200;
    c "start and STATUS poll of a 16-byte copy" Token;
  ]

let n = List.length cases

let slots =
  L.Placement.
    [
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int (4 * ((5 * n) + 1)));
      L.Slot.make ~home:Ram ~align:32L In
        (Int64.of_int (D.Descriptor.size * 2 * n));
      L.Slot.make ~home:Ram ~align:64L In (Int64.of_int length);
      L.Slot.make ~home:Dtcm ~align:64L Out (Int64.of_int length);
      L.Slot.make ~home:Dtcm ~align:64L In 256L;
      L.Slot.make ~home:Dtcm ~align:16L Out (Int64.of_int (16 * n));
      L.Slot.make ~home:Ram ~align:16L In (Int64.of_int (16 * n));
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int (16 * n));
      L.Slot.make ~home:Ram ~align:16L In 16L;
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int (16 * n));
    ]

let source =
  String.init length (fun k -> Char.chr (((k * 7) + (k / 256)) land 0xFF))

let expected = source

let inputs p (l : L.t) =
  let address k = (List.nth l.slots k).L.Placed_slot.address in
  let off a k = Int64.add a (Int64.of_int k) in
  let src = address 2 and dst = address dst_slot in
  let sram_flag k = off (address 6) (16 * k)
  and dtcm_flag k = off (address 7) (16 * k)
  and token k = off (address 9) (16 * k) in
  let copy = D.Descriptor.copy ~src ~dst (Int64.of_int length) in
  let always address = { D.Poll.address; mask = 0l; value = 0l } in
  let open Err.Syntax in
  let* chains, table, _ =
    Err.List.fold_left
      (fun (chains, table, at) (k, (c : Case.t)) ->
        let flag : D.Poll.t option =
          match c.flag with
          | No_flag -> None
          | Sram -> Some { address = sram_flag k; mask = 1l; value = 1l }
          | Dtcm -> Some { address = dtcm_flag k; mask = 1l; value = 1l }
        in
        (* The gate copies a beat of the source onto itself in SRAM. *)
        let gate () = D.Descriptor.copy ?poll:flag ~src ~dst:src 16L in
        let to_token = D.Descriptor.copy ~src:(address 8) ~dst:(token k) 16L in
        let descriptors =
          match c.chain with
          | No_chain -> []
          | Copy -> [ copy ]
          | Gated -> [ gate (); copy ]
          | Handshake -> [ gate (); to_token ]
          | Polled_copy `Sram -> [ { copy with poll = Some (always src) } ]
          | Polled_copy `Dtcm -> [ { copy with poll = Some (always dst) } ]
          | Token -> [ to_token ]
        in
        let flag_address =
          match c.flag with
          | No_flag -> 0
          | Sram -> Int64.to_int (sram_flag k)
          | Dtcm -> Int64.to_int (dtcm_flag k)
        in
        let row chain =
          [
            (if c.token then Int64.to_int (token k) else 0);
            c.work;
            c.delay;
            flag_address;
            chain;
          ]
        in
        match descriptors with
        | [] -> Err.return (chains, row 0 @ table, at)
        | _ ->
            let* bytes = D.chain p ~at descriptors in
            Err.return
              ( bytes :: chains,
                row (Int64.to_int at) @ table,
                off at (String.length bytes) ))
      ([], [], address 1)
      (List.mapi (fun k c -> (k, c)) cases)
  in
  Err.return
    [
      Dma_copy.words (n :: List.rev table);
      String.concat "" (List.rev chains);
      source;
      "";
      String.make 256 '\001';
      "";
      "";
      "";
      String.make 16 '\001';
      "";
    ]

let andi = Dma_copy.andi
let beqz = Dma_copy.beqz
let fence = Dma_copy.fence
let mcycle = Dma_copy.mcycle
let sub = Dma_copy.sub

(* [a1] the cases, [a2] the timings, [a3] the cases left, [a4] the DMA's
   registers, [a5] its start word, [a6] the work buffer; per case [t5] the
   chain, [t3] a start. *)
let items () =
  let open Err.Syntax in
  let open Gpr in
  let l = Label.named in
  let e8 = Vtype.Sew.E8 in
  let* base = Insn.load_imm a4 (Int64.to_int D.base) in
  Ok
    (start_items
    @ [
        lw a1 a0 0;
        lw a2 a0 (4 * timings_slot);
        lw a6 a0 16;
        addi t1 a6 128;
        lw a3 a1 0;
        addi a1 a1 4;
      ]
    @ List.map Item.insn base
    @ [
        li a5 D.ctrl_start;
        vset_item 8 8;
        Item.label (l "case");
        lw t5 a1 0;
        lw t6 a1 12;
        fence;
        mcycle t3;
        beqz t5 (l "no_start");
        sw t5 a4 (D.Register.offset Desc_addr);
        sw a5 a4 (D.Register.offset Ctrl);
        Item.label (l "no_start");
        beqz t6 (l "no_work");
      ]
    @ body_items
        [
          Item.label (l "trip");
          vle e8 Vreg.v0 a6;
          vle e8 Vreg.v8 t1;
          addi t6 t6 (-1);
          bnez t6 (l "trip");
        ]
    @ [
        Item.label (l "no_work");
        fence;
        mcycle t4;
        sub t4 t4 t3;
        sw t4 a2 0;
        lw t0 a4 (D.Register.offset Cur_desc);
        sw t0 a2 8;
        lw t6 a1 8;
        beqz t6 (l "no_delay");
        Item.label (l "delay");
        addi t6 t6 (-1);
        bnez t6 (l "delay");
        Item.label (l "no_delay");
        lw t0 a1 4;
        li t2 1;
        fence;
        mcycle t3;
        beqz t0 (l "no_flag");
        sw t2 t0 0;
        Item.label (l "no_flag");
        lw t0 a1 16;
        beqz t0 (l "status");
        Item.label (l "token");
        lw t2 t0 0;
        beqz t2 (l "token");
        Item.insn (J { target = l "waited" });
        Item.label (l "status");
        beqz t5 (l "waited");
        Item.label (l "poll");
        lw t2 a4 (D.Register.offset Status);
        andi t4 t2 D.status_done;
        beqz t4 (l "poll");
        Item.label (l "waited");
        mcycle t4;
        sub t4 t4 t3;
        sw t4 a2 4;
        (* the token is in: the chain's [STATUS] settles after it *)
        Item.label (l "settle");
        beqz t5 (l "settled");
        lw t2 a4 (D.Register.offset Status);
        andi t4 t2 D.status_done;
        beqz t4 (l "settle");
        Item.label (l "settled");
        sw t2 a2 12;
        addi a2 a2 16;
        addi a1 a1 20;
        addi a3 a3 (-1);
        bnez a3 (l "case");
        li a0 0;
        Item.insn Ret;
      ])

let name = "poll"
