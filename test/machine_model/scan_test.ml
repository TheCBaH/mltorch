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
      reference checked: 111 instructions; 0 register moves, 138 stores, 167 loads, 0 slot moves, 0 rematerialized; 117 slots (784 bytes)
      linear scan checked: 111 instructions; 28 register moves, 67 stores, 53 loads, 0 slot moves, 9 rematerialized; 23 slots (148 bytes)
    aarch64, 3 registers chain/1:
      reference checked: 68 instructions; 0 register moves, 82 stores, 94 loads, 0 slot moves, 0 rematerialized; 73 slots (524 bytes)
      linear scan checked: 68 instructions; 16 register moves, 20 stores, 17 loads, 0 slot moves, 5 rematerialized; 9 slots (60 bytes)
    aarch64, 3 registers chain/2:
      reference checked: 41 instructions; 0 register moves, 56 stores, 65 loads, 0 slot moves, 0 rematerialized; 46 slots (324 bytes)
      linear scan checked: 41 instructions; 8 register moves, 13 stores, 11 loads, 0 slot moves, 5 rematerialized; 7 slots (44 bytes)
    aarch64, 3 registers bmm/0:
      reference checked: 65 instructions; 0 register moves, 88 stores, 100 loads, 0 slot moves, 0 rematerialized; 74 slots (536 bytes)
      linear scan checked: 65 instructions; 14 register moves, 27 stores, 24 loads, 0 slot moves, 7 rematerialized; 12 slots (80 bytes)
    aarch64, 3 registers softmax/0:
      reference checked: 205 instructions; 0 register moves, 180 stores, 248 loads, 0 slot moves, 0 rematerialized; 162 slots (1140 bytes)
      linear scan checked: 205 instructions; 19 register moves, 48 stores, 41 loads, 0 slot moves, 14 rematerialized; 17 slots (120 bytes)
    aarch64, 3 registers sdpa/0:
      reference checked: 559 instructions; 0 register moves, 442 stores, 647 loads, 0 slot moves, 0 rematerialized; 407 slots (2896 bytes)
      linear scan checked: 559 instructions; 45 register moves, 129 stores, 108 loads, 0 slot moves, 24 rematerialized; 29 slots (212 bytes)
    aarch64, 3 registers conv/0:
      reference checked: 132 instructions; 0 register moves, 164 stores, 197 loads, 0 slot moves, 0 rematerialized; 140 slots (940 bytes)
      linear scan checked: 132 instructions; 21 register moves, 73 stores, 59 loads, 0 slot moves, 14 rematerialized; 26 slots (168 bytes)
    x86_64, 3 registers chain/0:
      reference checked: 116 instructions; 0 register moves, 143 stores, 172 loads, 0 slot moves, 0 rematerialized; 122 slots (816 bytes)
      linear scan checked: 116 instructions; 24 register moves, 61 stores, 61 loads, 0 slot moves, 14 rematerialized; 23 slots (148 bytes)
    x86_64, 3 registers chain/1:
      reference checked: 73 instructions; 0 register moves, 87 stores, 99 loads, 0 slot moves, 0 rematerialized; 78 slots (540 bytes)
      linear scan checked: 73 instructions; 11 register moves, 22 stores, 23 loads, 0 slot moves, 7 rematerialized; 11 slots (76 bytes)
    x86_64, 3 registers chain/2:
      reference checked: 45 instructions; 0 register moves, 60 stores, 69 loads, 0 slot moves, 0 rematerialized; 50 slots (348 bytes)
      linear scan checked: 45 instructions; 13 register moves, 16 stores, 18 loads, 0 slot moves, 7 rematerialized; 7 slots (44 bytes)
    x86_64, 3 registers bmm/0:
      reference checked: 65 instructions; 0 register moves, 88 stores, 100 loads, 0 slot moves, 0 rematerialized; 74 slots (524 bytes)
      linear scan checked: 65 instructions; 17 register moves, 29 stores, 32 loads, 0 slot moves, 10 rematerialized; 12 slots (80 bytes)
    x86_64, 3 registers softmax/0:
      reference checked: 230 instructions; 0 register moves, 202 stores, 280 loads, 0 slot moves, 0 rematerialized; 184 slots (1300 bytes)
      linear scan checked: 230 instructions; 27 register moves, 41 stores, 48 loads, 0 slot moves, 28 rematerialized; 15 slots (104 bytes)
    x86_64, 3 registers sdpa/0:
      reference checked: 601 instructions; 0 register moves, 474 stores, 689 loads, 0 slot moves, 0 rematerialized; 439 slots (3132 bytes)
      linear scan checked: 601 instructions; 44 register moves, 109 stores, 115 loads, 0 slot moves, 53 rematerialized; 26 slots (188 bytes)
    x86_64, 3 registers conv/0:
      reference checked: 130 instructions; 0 register moves, 162 stores, 194 loads, 0 slot moves, 0 rematerialized; 138 slots (920 bytes)
      linear scan checked: 130 instructions; 31 register moves, 67 stores, 69 loads, 0 slot moves, 22 rematerialized; 26 slots (168 bytes)
    aarch64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb12: d2 does not hold %36
    aarch64, 3 registers hole: NOT CAUGHT
    aarch64, 3 registers live tie: NOT CAUGHT
    aarch64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb1: target constraint: an undeclared slot
    aarch64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb1: x2 does not hold %188
    x86_64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb12: xmm2.d does not hold %36
    x86_64, 3 registers hole: NOT CAUGHT
    x86_64, 3 registers live tie: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb6 i278: target constraint: an instruction operand in memory
    x86_64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb1: target constraint: an undeclared slot
    x86_64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb1: rdi does not hold %193 |}]

let%expect_test "both allocators pass the same checker; what each costs" =
  A64.compare ();
  X64.compare ();
  [%expect
    {|
    aarch64 chain/0:
      reference checked: 111 instructions; 0 register moves, 138 stores, 167 loads, 0 slot moves, 0 rematerialized; 117 slots (784 bytes)
      linear scan checked: 111 instructions; 34 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1:
      reference checked: 68 instructions; 0 register moves, 82 stores, 94 loads, 0 slot moves, 0 rematerialized; 73 slots (524 bytes)
      linear scan checked: 68 instructions; 16 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2:
      reference checked: 41 instructions; 0 register moves, 56 stores, 65 loads, 0 slot moves, 0 rematerialized; 46 slots (324 bytes)
      linear scan checked: 41 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0:
      reference checked: 65 instructions; 0 register moves, 88 stores, 100 loads, 0 slot moves, 0 rematerialized; 74 slots (536 bytes)
      linear scan checked: 65 instructions; 18 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0:
      reference checked: 205 instructions; 0 register moves, 180 stores, 248 loads, 0 slot moves, 0 rematerialized; 162 slots (1140 bytes)
      linear scan checked: 205 instructions; 54 register moves, 6 stores, 6 loads, 0 slot moves, 0 rematerialized; 6 slots (40 bytes)
    aarch64 sdpa/0:
      reference checked: 559 instructions; 0 register moves, 442 stores, 647 loads, 0 slot moves, 0 rematerialized; 407 slots (2896 bytes)
      linear scan checked: 559 instructions; 101 register moves, 30 stores, 27 loads, 0 slot moves, 0 rematerialized; 14 slots (100 bytes)
    aarch64 conv/0:
      reference checked: 132 instructions; 0 register moves, 164 stores, 197 loads, 0 slot moves, 0 rematerialized; 140 slots (940 bytes)
      linear scan checked: 132 instructions; 39 register moves, 3 stores, 3 loads, 0 slot moves, 3 rematerialized; 3 slots (24 bytes)
    x86_64 chain/0:
      reference checked: 116 instructions; 0 register moves, 143 stores, 172 loads, 0 slot moves, 0 rematerialized; 122 slots (816 bytes)
      linear scan checked: 116 instructions; 52 register moves, 18 stores, 16 loads, 0 slot moves, 3 rematerialized; 13 slots (84 bytes)
    x86_64 chain/1:
      reference checked: 73 instructions; 0 register moves, 87 stores, 99 loads, 0 slot moves, 0 rematerialized; 78 slots (540 bytes)
      linear scan checked: 73 instructions; 25 register moves, 1 stores, 1 loads, 0 slot moves, 1 rematerialized; 1 slots (8 bytes)
    x86_64 chain/2:
      reference checked: 45 instructions; 0 register moves, 60 stores, 69 loads, 0 slot moves, 0 rematerialized; 50 slots (348 bytes)
      linear scan checked: 45 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0:
      reference checked: 65 instructions; 0 register moves, 88 stores, 100 loads, 0 slot moves, 0 rematerialized; 74 slots (524 bytes)
      linear scan checked: 65 instructions; 31 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
    x86_64 softmax/0:
      reference checked: 230 instructions; 0 register moves, 202 stores, 280 loads, 0 slot moves, 0 rematerialized; 184 slots (1300 bytes)
      linear scan checked: 230 instructions; 58 register moves, 23 stores, 23 loads, 0 slot moves, 2 rematerialized; 12 slots (84 bytes)
    x86_64 sdpa/0:
      reference checked: 601 instructions; 0 register moves, 474 stores, 689 loads, 0 slot moves, 0 rematerialized; 439 slots (3132 bytes)
      linear scan checked: 601 instructions; 93 register moves, 50 stores, 49 loads, 0 slot moves, 13 rematerialized; 20 slots (148 bytes)
    x86_64 conv/0:
      reference checked: 130 instructions; 0 register moves, 162 stores, 194 loads, 0 slot moves, 0 rematerialized; 138 slots (920 bytes)
      linear scan checked: 130 instructions; 59 register moves, 21 stores, 19 loads, 0 slot moves, 3 rematerialized; 16 slots (104 bytes) |}]

let%expect_test "linear-scan mutations with full pools" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: w0 does not hold %30
    aarch64 hole: NOT CAUGHT
    aarch64 live tie: NOT CAUGHT
    aarch64 needed eviction: NOT CAUGHT
    aarch64 split move: caught on 2 kernels; first softmax/0: checker: fn0 bb20: x0 does not hold %52
    x86_64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: eax does not hold %30
    x86_64 hole: caught on 3 kernels; first chain/0: checker: fn0 bb13: r9 does not hold %225
    x86_64 live tie: caught on 7 kernels; first chain/0: checker: fn0 bb4: r12d does not hold %41
    x86_64 needed eviction: caught on 1 kernels; first sdpa/0: physical verifier: allocated fn0 bb6 i555: target constraint: an instruction operand in memory
    x86_64 split move: caught on 5 kernels; first chain/0: checker: fn0 bb6: ebx does not hold %2 |}]
