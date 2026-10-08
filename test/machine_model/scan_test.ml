(* M10.1: split linear scan against the reference allocator on the model
   kernels of both targets — both pass the same checker, and what each costs
   — and which kernels catch each linear-scan mutation. *)

open Machine_ir
module M = Machine_model.Mir_model
module F = Native_test.Graph_fixtures

let kernels () =
  let one name g = (name, g) in
  [
    one "chain" (F.chain ());
    one "bmm"
      (F.build "bmm"
         Graph_builder.(
           let* a = input ~shape:(F.s 1 1 1 2 5 7) () in
           let* b = input ~shape:(F.s 1 1 1 2 7 3) () in
           bmm a b));
    one "softmax"
      (F.build "softmax"
         Graph_builder.(
           let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
           softmax { Reduce.Softmax.axis = Axis.C } x));
    one "sdpa" (Routes_test.sdpa ());
    (* a padded, strided 3x3 convolution: enough loops and addresses to
       exhaust the full x86-64 pools *)
    one "conv"
      (F.build "conv"
         Graph_builder.(
           let* x = input ~shape:(F.s 2 1 1 7 7 3) () in
           let* w = constant ~shape:(F.s 4 1 1 3 3 3) () in
           let* bias = constant ~shape:(F.s1c 4) () in
           let axis = F.conv_axis ~kernel:3 ~stride:2 ~pad:1 in
           conv2d
             {
               Conv.Conv2d.h = axis;
               w = axis;
               in_channels = Dim.extent 3;
               groups = Op_config.Pos.of_int 1;
             }
             ~x ~weight:w ~bias ()));
  ]
  |> List.concat_map (fun (name, g) ->
      let b = Model_test.bundle g in
      let m =
        Result.get_ok (M.prepare ~pipeline:Ssa_backends.Pipeline.Exact b)
      in
      List.mapi
        (fun k gen ->
          ( Fmt.str "%s/%d" name k,
            gen,
            M.sites (List.nth b.Loop_ir.Loop_bundle.invocations k) ))
        (M.generic m))

module Run
    (T : Mir_sel.TARGET)
    (R : Machine_alloc.Mir_linear_scan.POOL)
    (X : sig
      val name : string

      val select :
        sites:Mir_failure.Site_entry.t array ->
        Mir_verify.Generic.t ->
        Mir_sel.Make(T).Verified.t
    end) =
struct
  module A = Machine_alloc.Mir_ref_alloc.Make (T) (R)
  module Ls = Machine_alloc.Mir_linear_scan.Make (T) (R)
  module V = Mir_phys_verify.Make (T)
  module C = Machine_check.Mir_checker.Make (T)

  let verdict v phys =
    match Err.payload (V.verify phys) with
    | Error d -> Fmt.str "physical verifier: %a" Mir_diagnostic.pp d
    | Ok phys -> (
        match Err.payload (C.check v phys) with
        | Error e -> Fmt.str "checker: %a" Machine_check.Mir_checker.pp_error e
        | Ok () -> "checked")

  let compare () =
    List.iter
      (fun (name, gen, sites) ->
        let v = X.select ~sites gen in
        let r = A.allocate v and l = Ls.allocate v in
        Fmt.pr "%s %s:@.  reference %s: %a@.  linear scan %s: %a@." X.name name
          (verdict v r) Machine_alloc.Mir_alloc_stats.pp
          (Machine_alloc.Mir_alloc_stats.of_program r)
          (verdict v l) Machine_alloc.Mir_alloc_stats.pp
          (Machine_alloc.Mir_alloc_stats.of_program l))
      (kernels ())

  let mutations () =
    List.iter
      (fun (label, mutation) ->
        let caught =
          List.filter_map
            (fun (name, gen, sites) ->
              let v = X.select ~sites gen in
              match verdict v (Ls.allocate ~mutation v) with
              | "checked" -> None
              | e -> Some (name ^ ": " ^ e))
            (kernels ())
        in
        Fmt.pr "%s %s: %s@." X.name label
          (match caught with
          | [] -> "NOT CAUGHT"
          | first :: _ ->
              Fmt.str "caught on %d kernels; first %s" (List.length caught)
                first))
      Machine_alloc.Mir_linear_scan.Mutation.
        [
          ("call interval", Call_interval);
          ("hole", Hole);
          ("live tie", Live_tie);
          ("needed eviction", Needed_eviction);
          ("split move", Split_move);
        ]
end

module A64_select_ = struct
  let select ~sites g =
    (Result.get_ok
       (Err.payload
          (Machine_target_aarch64.A64_select.program ~sites
             ~unlisted:Mir_failure.Unlisted.Unreachable g)))
      .Machine_target_aarch64.A64_select.selected
end

module X64_select_ = struct
  let select ~sites g =
    (Result.get_ok
       (Err.payload
          (Machine_target_x86_64.X64_select.program ~sites
             ~unlisted:Mir_failure.Unlisted.Unreachable g)))
      .Machine_target_x86_64.X64_select.selected
end

module A64 =
  Run (Machine_target_aarch64.A64) (Machine_target_aarch64.A64_regs)
    (struct
      let name = "aarch64"

      let select ~sites g =
        (Result.get_ok
           (Err.payload
              (Machine_target_aarch64.A64_select.program ~sites
                 ~unlisted:Mir_failure.Unlisted.Unreachable g)))
          .Machine_target_aarch64.A64_select.selected
    end)

module X64 =
  Run (Machine_target_x86_64.X64) (Machine_target_x86_64.X64_regs)
    (struct
      let name = "x86_64"

      let select ~sites g =
        (Result.get_ok
           (Err.payload
              (Machine_target_x86_64.X64_select.program ~sites
                 ~unlisted:Mir_failure.Unlisted.Unreachable g)))
          .Machine_target_x86_64.X64_select.selected
    end)

(* Three registers a bank: pressure on every kernel, so splitting, spilling,
   eviction and lifetime holes all decide something. *)
let take n l = List.filteri (fun i _ -> i < n) l

module A64_small =
  Run
    (Machine_target_aarch64.A64)
    (struct
      include Machine_target_aarch64.A64_regs

      let allocatable bank = take 3 (allocatable bank)
    end)
    (struct
      include A64_select_

      let name = "aarch64, 3 registers"
    end)

module X64_small =
  Run
    (Machine_target_x86_64.X64)
    (struct
      include Machine_target_x86_64.X64_regs

      let allocatable bank = take 3 (allocatable bank)
    end)
    (struct
      include X64_select_

      let name = "x86_64, 3 registers"
    end)

(* The hole mutation is caught under pressure on both targets, and with full
   pools on x86-64 only by the convolution: its loop exits reload values while
   every other register is taken, into one an interval holds across a lifetime
   hole. AArch64's larger pools always leave one free there. A needed eviction
   is caught under pressure only: with full pools no eviction meets an operand
   of the instruction that evicts. *)
let%expect_test "under pressure: both checked; what each mutation breaks" =
  A64_small.compare ();
  X64_small.compare ();
  A64_small.mutations ();
  X64_small.mutations ();
  [%expect
    {|
    aarch64, 3 registers chain/0:
      reference checked: 114 instructions; 0 register moves, 117 stores, 146 loads, 0 slot moves, 0 rematerialized; 108 slots (696 bytes)
      linear scan checked: 114 instructions; 11 register moves, 47 stores, 38 loads, 0 slot moves, 5 rematerialized; 14 slots (76 bytes)
    aarch64, 3 registers chain/1:
      reference checked: 88 instructions; 0 register moves, 90 stores, 102 loads, 0 slot moves, 0 rematerialized; 87 slots (612 bytes)
      linear scan checked: 88 instructions; 9 register moves, 15 stores, 12 loads, 0 slot moves, 2 rematerialized; 5 slots (32 bytes)
    aarch64, 3 registers chain/2:
      reference checked: 45 instructions; 0 register moves, 48 stores, 57 loads, 0 slot moves, 0 rematerialized; 44 slots (300 bytes)
      linear scan checked: 45 instructions; 8 register moves, 12 stores, 10 loads, 0 slot moves, 2 rematerialized; 4 slots (24 bytes)
    aarch64, 3 registers bmm/0:
      reference checked: 65 instructions; 0 register moves, 70 stores, 82 loads, 0 slot moves, 0 rematerialized; 65 slots (452 bytes)
      linear scan checked: 65 instructions; 11 register moves, 20 stores, 17 loads, 0 slot moves, 3 rematerialized; 7 slots (44 bytes)
    aarch64, 3 registers softmax/0:
      reference checked: 258 instructions; 0 register moves, 213 stores, 285 loads, 0 slot moves, 0 rematerialized; 205 slots (1468 bytes)
      linear scan checked: 258 instructions; 17 register moves, 44 stores, 40 loads, 0 slot moves, 12 rematerialized; 12 slots (80 bytes)
    aarch64, 3 registers sdpa/0:
      reference checked: 629 instructions; 0 register moves, 472 stores, 677 loads, 0 slot moves, 0 rematerialized; 457 slots (3276 bytes)
      linear scan checked: 629 instructions; 30 register moves, 88 stores, 76 loads, 0 slot moves, 19 rematerialized; 16 slots (108 bytes)
    aarch64, 3 registers conv/0:
      reference checked: 135 instructions; 0 register moves, 139 stores, 172 loads, 0 slot moves, 0 rematerialized; 129 slots (836 bytes)
      linear scan checked: 135 instructions; 12 register moves, 54 stores, 45 loads, 0 slot moves, 8 rematerialized; 15 slots (80 bytes)
    x86_64, 3 registers chain/0:
      reference checked: 98 instructions; 0 register moves, 101 stores, 130 loads, 0 slot moves, 0 rematerialized; 92 slots (576 bytes)
      linear scan checked: 98 instructions; 14 register moves, 47 stores, 44 loads, 0 slot moves, 8 rematerialized; 14 slots (76 bytes)
    x86_64, 3 registers chain/1:
      reference checked: 63 instructions; 0 register moves, 65 stores, 77 loads, 0 slot moves, 0 rematerialized; 62 slots (412 bytes)
      linear scan checked: 63 instructions; 9 register moves, 15 stores, 14 loads, 0 slot moves, 4 rematerialized; 7 slots (44 bytes)
    x86_64, 3 registers chain/2:
      reference checked: 35 instructions; 0 register moves, 38 stores, 47 loads, 0 slot moves, 0 rematerialized; 34 slots (220 bytes)
      linear scan checked: 35 instructions; 9 register moves, 11 stores, 10 loads, 0 slot moves, 4 rematerialized; 5 slots (28 bytes)
    x86_64, 3 registers bmm/0:
      reference checked: 52 instructions; 0 register moves, 57 stores, 69 loads, 0 slot moves, 0 rematerialized; 52 slots (348 bytes)
      linear scan checked: 52 instructions; 10 register moves, 17 stores, 17 loads, 0 slot moves, 6 rematerialized; 8 slots (48 bytes)
    x86_64, 3 registers softmax/0:
      reference checked: 232 instructions; 0 register moves, 184 stores, 266 loads, 0 slot moves, 0 rematerialized; 176 slots (1236 bytes)
      linear scan checked: 232 instructions; 23 register moves, 37 stores, 39 loads, 0 slot moves, 21 rematerialized; 10 slots (64 bytes)
    x86_64, 3 registers sdpa/0:
      reference checked: 573 instructions; 0 register moves, 406 stores, 621 loads, 0 slot moves, 0 rematerialized; 391 slots (2748 bytes)
      linear scan checked: 573 instructions; 38 register moves, 70 stores, 71 loads, 0 slot moves, 43 rematerialized; 12 slots (76 bytes)
    x86_64, 3 registers conv/0:
      reference checked: 114 instructions; 0 register moves, 118 stores, 150 loads, 0 slot moves, 0 rematerialized; 108 slots (680 bytes)
      linear scan checked: 114 instructions; 15 register moves, 52 stores, 52 loads, 0 slot moves, 9 rematerialized; 16 slots (84 bytes)
    aarch64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb12: d2 does not hold %36
    aarch64, 3 registers hole: caught on 7 kernels; first chain/0: checker: fn0 bb7: w2 does not hold %6
    aarch64, 3 registers live tie: NOT CAUGHT
    aarch64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb6 i233: target constraint: an instruction operand in memory
    aarch64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb4: w2 does not hold %35
    x86_64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb12: xmm2.d does not hold %36
    x86_64, 3 registers hole: caught on 7 kernels; first chain/0: checker: fn0 bb7: edi does not hold %6
    x86_64, 3 registers live tie: caught on 7 kernels; first chain/0: checker: fn0 bb4: eax does not hold %41
    x86_64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb4 i217: target constraint: an instruction operand in memory
    x86_64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb4: edi does not hold %35 |}]

let%expect_test "both allocators pass the same checker; what each costs" =
  A64.compare ();
  X64.compare ();
  [%expect
    {|
    aarch64 chain/0:
      reference checked: 114 instructions; 0 register moves, 117 stores, 146 loads, 0 slot moves, 0 rematerialized; 108 slots (696 bytes)
      linear scan checked: 114 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1:
      reference checked: 88 instructions; 0 register moves, 90 stores, 102 loads, 0 slot moves, 0 rematerialized; 87 slots (612 bytes)
      linear scan checked: 88 instructions; 6 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2:
      reference checked: 45 instructions; 0 register moves, 48 stores, 57 loads, 0 slot moves, 0 rematerialized; 44 slots (300 bytes)
      linear scan checked: 45 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0:
      reference checked: 65 instructions; 0 register moves, 70 stores, 82 loads, 0 slot moves, 0 rematerialized; 65 slots (452 bytes)
      linear scan checked: 65 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0:
      reference checked: 258 instructions; 0 register moves, 213 stores, 285 loads, 0 slot moves, 0 rematerialized; 205 slots (1468 bytes)
      linear scan checked: 258 instructions; 17 register moves, 3 stores, 3 loads, 0 slot moves, 0 rematerialized; 1 slots (4 bytes)
    aarch64 sdpa/0:
      reference checked: 629 instructions; 0 register moves, 472 stores, 677 loads, 0 slot moves, 0 rematerialized; 457 slots (3276 bytes)
      linear scan checked: 629 instructions; 30 register moves, 3 stores, 2 loads, 0 slot moves, 0 rematerialized; 1 slots (8 bytes)
    aarch64 conv/0:
      reference checked: 135 instructions; 0 register moves, 139 stores, 172 loads, 0 slot moves, 0 rematerialized; 129 slots (836 bytes)
      linear scan checked: 135 instructions; 19 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 chain/0:
      reference checked: 98 instructions; 0 register moves, 101 stores, 130 loads, 0 slot moves, 0 rematerialized; 92 slots (576 bytes)
      linear scan checked: 98 instructions; 26 register moves, 4 stores, 4 loads, 0 slot moves, 2 rematerialized; 4 slots (20 bytes)
    x86_64 chain/1:
      reference checked: 63 instructions; 0 register moves, 65 stores, 77 loads, 0 slot moves, 0 rematerialized; 62 slots (412 bytes)
      linear scan checked: 63 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 chain/2:
      reference checked: 35 instructions; 0 register moves, 38 stores, 47 loads, 0 slot moves, 0 rematerialized; 34 slots (220 bytes)
      linear scan checked: 35 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0:
      reference checked: 52 instructions; 0 register moves, 57 stores, 69 loads, 0 slot moves, 0 rematerialized; 52 slots (348 bytes)
      linear scan checked: 52 instructions; 13 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 softmax/0:
      reference checked: 232 instructions; 0 register moves, 184 stores, 266 loads, 0 slot moves, 0 rematerialized; 176 slots (1236 bytes)
      linear scan checked: 232 instructions; 38 register moves, 21 stores, 20 loads, 0 slot moves, 3 rematerialized; 7 slots (44 bytes)
    x86_64 sdpa/0:
      reference checked: 573 instructions; 0 register moves, 406 stores, 621 loads, 0 slot moves, 0 rematerialized; 391 slots (2748 bytes)
      linear scan checked: 573 instructions; 55 register moves, 38 stores, 35 loads, 0 slot moves, 13 rematerialized; 8 slots (52 bytes)
    x86_64 conv/0:
      reference checked: 114 instructions; 0 register moves, 118 stores, 150 loads, 0 slot moves, 0 rematerialized; 108 slots (680 bytes)
      linear scan checked: 114 instructions; 30 register moves, 12 stores, 12 loads, 0 slot moves, 2 rematerialized; 6 slots (28 bytes) |}]

let%expect_test "linear-scan mutations with full pools" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: w0 does not hold %30
    aarch64 hole: NOT CAUGHT
    aarch64 live tie: NOT CAUGHT
    aarch64 needed eviction: NOT CAUGHT
    aarch64 split move: caught on 2 kernels; first softmax/0: checker: fn0 bb6: w20 does not hold %2
    x86_64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: eax does not hold %30
    x86_64 hole: caught on 1 kernels; first conv/0: checker: fn0 bb15: esi does not hold %14
    x86_64 live tie: caught on 7 kernels; first chain/0: checker: fn0 bb4: r8d does not hold %41
    x86_64 needed eviction: caught on 2 kernels; first chain/0: physical verifier: allocated fn0 bb16 i296: target constraint: an instruction operand in memory
    x86_64 split move: caught on 4 kernels; first chain/0: checker: fn0 bb14: r8d does not hold %54 |}]
