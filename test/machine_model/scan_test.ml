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
      reference checked: 91 instructions; 0 register moves, 120 stores, 140 loads, 0 slot moves, 0 rematerialized; 98 slots (632 bytes)
      linear scan checked: 91 instructions; 11 register moves, 62 stores, 51 loads, 0 slot moves, 6 rematerialized; 25 slots (164 bytes)
    aarch64, 3 registers chain/1:
      reference checked: 58 instructions; 0 register moves, 80 stores, 82 loads, 0 slot moves, 0 rematerialized; 67 slots (476 bytes)
      linear scan checked: 58 instructions; 4 register moves, 39 stores, 30 loads, 0 slot moves, 3 rematerialized; 17 slots (124 bytes)
    aarch64, 3 registers chain/2:
      reference checked: 27 instructions; 0 register moves, 42 stores, 45 loads, 0 slot moves, 0 rematerialized; 32 slots (212 bytes)
      linear scan checked: 27 instructions; 11 register moves, 8 stores, 8 loads, 0 slot moves, 2 rematerialized; 7 slots (44 bytes)
    aarch64, 3 registers bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (320 bytes)
      linear scan checked: 38 instructions; 11 register moves, 23 stores, 19 loads, 0 slot moves, 4 rematerialized; 11 slots (76 bytes)
    aarch64, 3 registers softmax/0:
      reference checked: 180 instructions; 0 register moves, 155 stores, 213 loads, 0 slot moves, 0 rematerialized; 137 slots (940 bytes)
      linear scan checked: 180 instructions; 22 register moves, 38 stores, 36 loads, 0 slot moves, 8 rematerialized; 19 slots (136 bytes)
    aarch64, 3 registers sdpa/0:
      reference checked: 516 instructions; 0 register moves, 411 stores, 590 loads, 0 slot moves, 0 rematerialized; 370 slots (2596 bytes)
      linear scan checked: 516 instructions; 29 register moves, 130 stores, 108 loads, 0 slot moves, 16 rematerialized; 35 slots (256 bytes)
    aarch64, 3 registers conv/0:
      reference checked: 105 instructions; 0 register moves, 139 stores, 161 loads, 0 slot moves, 0 rematerialized; 114 slots (732 bytes)
      linear scan checked: 105 instructions; 11 register moves, 69 stores, 57 loads, 0 slot moves, 9 rematerialized; 29 slots (188 bytes)
    x86_64, 3 registers chain/0:
      reference checked: 92 instructions; 0 register moves, 121 stores, 141 loads, 0 slot moves, 0 rematerialized; 99 slots (632 bytes)
      linear scan checked: 92 instructions; 17 register moves, 61 stores, 57 loads, 0 slot moves, 11 rematerialized; 24 slots (156 bytes)
    x86_64, 3 registers chain/1:
      reference checked: 55 instructions; 0 register moves, 77 stores, 79 loads, 0 slot moves, 0 rematerialized; 64 slots (428 bytes)
      linear scan checked: 55 instructions; 5 register moves, 35 stores, 30 loads, 0 slot moves, 8 rematerialized; 13 slots (92 bytes)
    x86_64, 3 registers chain/2:
      reference checked: 27 instructions; 0 register moves, 42 stores, 45 loads, 0 slot moves, 0 rematerialized; 32 slots (204 bytes)
      linear scan checked: 27 instructions; 11 register moves, 9 stores, 11 loads, 0 slot moves, 2 rematerialized; 7 slots (44 bytes)
    x86_64, 3 registers bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (308 bytes)
      linear scan checked: 38 instructions; 12 register moves, 22 stores, 22 loads, 0 slot moves, 4 rematerialized; 12 slots (80 bytes)
    x86_64, 3 registers softmax/0:
      reference checked: 200 instructions; 0 register moves, 172 stores, 240 loads, 0 slot moves, 0 rematerialized; 154 slots (1060 bytes)
      linear scan checked: 200 instructions; 33 register moves, 39 stores, 41 loads, 0 slot moves, 16 rematerialized; 17 slots (120 bytes)
    x86_64, 3 registers sdpa/0:
      reference checked: 543 instructions; 0 register moves, 427 stores, 616 loads, 0 slot moves, 0 rematerialized; 386 slots (2704 bytes)
      linear scan checked: 543 instructions; 38 register moves, 111 stores, 98 loads, 0 slot moves, 41 rematerialized; 31 slots (224 bytes)
    x86_64, 3 registers conv/0:
      reference checked: 100 instructions; 0 register moves, 134 stores, 155 loads, 0 slot moves, 0 rematerialized; 109 slots (688 bytes)
      linear scan checked: 100 instructions; 15 register moves, 71 stores, 68 loads, 0 slot moves, 11 rematerialized; 27 slots (176 bytes)
    aarch64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb14: [slot8:8] does not hold %62
    aarch64, 3 registers hole: NOT CAUGHT
    aarch64, 3 registers live tie: NOT CAUGHT
    aarch64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb6 i236: target constraint: an instruction operand in memory
    aarch64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb1: [slot1:8] does not hold %202
    x86_64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb17: xmm1.d does not hold %115
    x86_64, 3 registers hole: NOT CAUGHT
    x86_64, 3 registers live tie: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb15 i297: target constraint: an instruction operand in memory
    x86_64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb4 i216: target constraint: an instruction operand in memory
    x86_64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb1: eax does not hold %32 |}]

let%expect_test "both allocators pass the same checker; what each costs" =
  A64.compare ();
  X64.compare ();
  [%expect
    {|
    aarch64 chain/0:
      reference checked: 91 instructions; 0 register moves, 120 stores, 140 loads, 0 slot moves, 0 rematerialized; 98 slots (632 bytes)
      linear scan checked: 91 instructions; 39 register moves, 1 stores, 1 loads, 0 slot moves, 2 rematerialized; 1 slots (8 bytes)
    aarch64 chain/1:
      reference checked: 58 instructions; 0 register moves, 80 stores, 82 loads, 0 slot moves, 0 rematerialized; 67 slots (476 bytes)
      linear scan checked: 58 instructions; 21 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2:
      reference checked: 27 instructions; 0 register moves, 42 stores, 45 loads, 0 slot moves, 0 rematerialized; 32 slots (212 bytes)
      linear scan checked: 27 instructions; 15 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (320 bytes)
      linear scan checked: 38 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0:
      reference checked: 180 instructions; 0 register moves, 155 stores, 213 loads, 0 slot moves, 0 rematerialized; 137 slots (940 bytes)
      linear scan checked: 180 instructions; 50 register moves, 4 stores, 4 loads, 0 slot moves, 3 rematerialized; 4 slots (28 bytes)
    aarch64 sdpa/0:
      reference checked: 516 instructions; 0 register moves, 411 stores, 590 loads, 0 slot moves, 0 rematerialized; 370 slots (2596 bytes)
      linear scan checked: 516 instructions; 122 register moves, 29 stores, 27 loads, 0 slot moves, 0 rematerialized; 14 slots (104 bytes)
    aarch64 conv/0:
      reference checked: 105 instructions; 0 register moves, 139 stores, 161 loads, 0 slot moves, 0 rematerialized; 114 slots (732 bytes)
      linear scan checked: 105 instructions; 46 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 5 slots (40 bytes)
    x86_64 chain/0:
      reference checked: 92 instructions; 0 register moves, 121 stores, 141 loads, 0 slot moves, 0 rematerialized; 99 slots (632 bytes)
      linear scan checked: 92 instructions; 45 register moves, 18 stores, 18 loads, 0 slot moves, 4 rematerialized; 14 slots (92 bytes)
    x86_64 chain/1:
      reference checked: 55 instructions; 0 register moves, 77 stores, 79 loads, 0 slot moves, 0 rematerialized; 64 slots (428 bytes)
      linear scan checked: 55 instructions; 23 register moves, 5 stores, 5 loads, 0 slot moves, 5 rematerialized; 4 slots (28 bytes)
    x86_64 chain/2:
      reference checked: 27 instructions; 0 register moves, 42 stores, 45 loads, 0 slot moves, 0 rematerialized; 32 slots (204 bytes)
      linear scan checked: 27 instructions; 19 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (308 bytes)
      linear scan checked: 38 instructions; 24 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
    x86_64 softmax/0:
      reference checked: 200 instructions; 0 register moves, 172 stores, 240 loads, 0 slot moves, 0 rematerialized; 154 slots (1060 bytes)
      linear scan checked: 200 instructions; 56 register moves, 23 stores, 23 loads, 0 slot moves, 5 rematerialized; 13 slots (96 bytes)
    x86_64 sdpa/0:
      reference checked: 543 instructions; 0 register moves, 427 stores, 616 loads, 0 slot moves, 0 rematerialized; 386 slots (2704 bytes)
      linear scan checked: 543 instructions; 90 register moves, 63 stores, 57 loads, 0 slot moves, 18 rematerialized; 24 slots (176 bytes)
    x86_64 conv/0:
      reference checked: 100 instructions; 0 register moves, 134 stores, 155 loads, 0 slot moves, 0 rematerialized; 109 slots (688 bytes)
      linear scan checked: 100 instructions; 59 register moves, 21 stores, 21 loads, 0 slot moves, 4 rematerialized; 17 slots (112 bytes) |}]

let%expect_test "linear-scan mutations with full pools" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: w0 does not hold %30
    aarch64 hole: NOT CAUGHT
    aarch64 live tie: NOT CAUGHT
    aarch64 needed eviction: NOT CAUGHT
    aarch64 split move: caught on 2 kernels; first softmax/0: checker: fn0 bb17: x19 does not hold %257
    x86_64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb10: xmm1.d does not hold %35
    x86_64 hole: caught on 4 kernels; first chain/0: checker: fn0 bb10: r14 does not hold %227
    x86_64 live tie: caught on 7 kernels; first chain/0: checker: fn0 bb4: r13d does not hold %41
    x86_64 needed eviction: caught on 4 kernels; first chain/0: physical verifier: allocated fn0 bb12 i278: target constraint: an instruction operand in memory
    x86_64 split move: caught on 5 kernels; first chain/0: checker: fn0 bb6: r9 does not hold %193 |}]
