(* M10.1: split linear scan against the reference allocator on the model
   kernels of both targets — both pass the same checker, and what each costs
   — and which kernels catch each linear-scan mutation. *)

open Machine_ir
module M = Machine_model.Mir_model
module F = Native_test.Graph_fixtures

(* The structured route is the default; [~frontend] lowers the same graphs
   through another route. The comparison route's kernels keep the shapes the
   scheduling evidence (sink moves, flags-writer motion) was first measured
   on, which the structured route's smaller kernels no longer exercise. *)
let kernels ?frontend () =
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
      let generic =
        match frontend with
        | None ->
            M.generic
              (Result.get_ok
                 (M.prepare ~pipeline:Ssa_backends.Pipeline.Exact b))
        | Some frontend ->
            List.map
              (fun inv ->
                let p =
                  Result.get_ok
                    (Ssa_backends.program ~frontend Ssa_backends.Pipeline.Exact
                       inv)
                in
                let planning =
                  Mir_planning.make
                    ~subject:(Machine_lower.Mir_lower.subject p)
                    ~policy:"reference_f64" ~schedule:"scalar"
                    ~precision:Mir_planning.Precision.F64
                    ~lanes:(Mir_type.Lanes.of_int 1)
                    ~fma:Mir_planning.Fma.Forbidden ~capabilities:[]
                in
                (Result.get_ok
                   (Err.payload
                      (Machine_lower.Mir_lower.program
                         ~planning:(Some planning) p)))
                  .Machine_lower.Mir_lower.program)
              b.Loop_ir.Loop_bundle.invocations
      in
      List.mapi
        (fun k gen ->
          ( Fmt.str "%s/%d" name k,
            gen,
            M.sites (List.nth b.Loop_ir.Loop_bundle.invocations k) ))
        generic)

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
      reference checked: 30 instructions; 0 register moves, 42 stores, 46 loads, 0 slot moves, 0 rematerialized; 33 slots (216 bytes)
      linear scan checked: 30 instructions; 12 register moves, 9 stores, 9 loads, 0 slot moves, 2 rematerialized; 8 slots (52 bytes)
    aarch64, 3 registers bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (320 bytes)
      linear scan checked: 38 instructions; 11 register moves, 23 stores, 19 loads, 0 slot moves, 4 rematerialized; 11 slots (76 bytes)
    aarch64, 3 registers softmax/0:
      reference checked: 64 instructions; 0 register moves, 88 stores, 101 loads, 0 slot moves, 0 rematerialized; 70 slots (472 bytes)
      linear scan checked: 64 instructions; 17 register moves, 41 stores, 37 loads, 0 slot moves, 8 rematerialized; 18 slots (128 bytes)
    aarch64, 3 registers sdpa/0:
      reference checked: 127 instructions; 0 register moves, 184 stores, 214 loads, 0 slot moves, 0 rematerialized; 144 slots (1020 bytes)
      linear scan checked: 127 instructions; 27 register moves, 115 stores, 94 loads, 0 slot moves, 18 rematerialized; 28 slots (204 bytes)
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
      reference checked: 36 instructions; 0 register moves, 48 stores, 53 loads, 0 slot moves, 0 rematerialized; 39 slots (256 bytes)
      linear scan checked: 36 instructions; 15 register moves, 12 stores, 14 loads, 0 slot moves, 2 rematerialized; 9 slots (60 bytes)
    x86_64, 3 registers bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (308 bytes)
      linear scan checked: 38 instructions; 12 register moves, 22 stores, 22 loads, 0 slot moves, 4 rematerialized; 12 slots (80 bytes)
    x86_64, 3 registers softmax/0:
      reference checked: 85 instructions; 0 register moves, 109 stores, 132 loads, 0 slot moves, 0 rematerialized; 91 slots (624 bytes)
      linear scan checked: 85 instructions; 28 register moves, 41 stores, 43 loads, 0 slot moves, 14 rematerialized; 17 slots (120 bytes)
    x86_64, 3 registers sdpa/0:
      reference checked: 157 instructions; 0 register moves, 213 stores, 253 loads, 0 slot moves, 0 rematerialized; 173 slots (1232 bytes)
      linear scan checked: 157 instructions; 40 register moves, 105 stores, 94 loads, 0 slot moves, 40 rematerialized; 27 slots (192 bytes)
    x86_64, 3 registers conv/0:
      reference checked: 100 instructions; 0 register moves, 134 stores, 155 loads, 0 slot moves, 0 rematerialized; 109 slots (688 bytes)
      linear scan checked: 100 instructions; 15 register moves, 71 stores, 68 loads, 0 slot moves, 11 rematerialized; 27 slots (176 bytes)
    aarch64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb17: d1 does not hold %90
    aarch64, 3 registers hole: caught on 1 kernels; first sdpa/0: checker: fn0 bb29: w0 does not hold %22
    aarch64, 3 registers live tie: NOT CAUGHT
    aarch64, 3 registers needed eviction: caught on 7 kernels; first chain/0: physical verifier: allocated fn0 bb6 i236: target constraint: an instruction operand in memory
    aarch64, 3 registers split move: caught on 7 kernels; first chain/0: checker: fn0 bb1: [slot1:8] does not hold %202
    x86_64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb17: xmm1.d does not hold %90
    x86_64, 3 registers hole: caught on 1 kernels; first sdpa/0: checker: fn0 bb29: edi does not hold %22
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
      reference checked: 30 instructions; 0 register moves, 42 stores, 46 loads, 0 slot moves, 0 rematerialized; 33 slots (216 bytes)
      linear scan checked: 30 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (320 bytes)
      linear scan checked: 38 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0:
      reference checked: 64 instructions; 0 register moves, 88 stores, 101 loads, 0 slot moves, 0 rematerialized; 70 slots (472 bytes)
      linear scan checked: 64 instructions; 55 register moves, 5 stores, 5 loads, 0 slot moves, 6 rematerialized; 3 slots (24 bytes)
    aarch64 sdpa/0:
      reference checked: 127 instructions; 0 register moves, 184 stores, 214 loads, 0 slot moves, 0 rematerialized; 144 slots (1020 bytes)
      linear scan checked: 127 instructions; 97 register moves, 36 stores, 35 loads, 0 slot moves, 0 rematerialized; 13 slots (92 bytes)
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
      reference checked: 36 instructions; 0 register moves, 48 stores, 53 loads, 0 slot moves, 0 rematerialized; 39 slots (256 bytes)
      linear scan checked: 36 instructions; 19 register moves, 1 stores, 1 loads, 0 slot moves, 1 rematerialized; 1 slots (8 bytes)
    x86_64 bmm/0:
      reference checked: 38 instructions; 0 register moves, 61 stores, 64 loads, 0 slot moves, 0 rematerialized; 47 slots (308 bytes)
      linear scan checked: 38 instructions; 24 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
    x86_64 softmax/0:
      reference checked: 85 instructions; 0 register moves, 109 stores, 132 loads, 0 slot moves, 0 rematerialized; 91 slots (624 bytes)
      linear scan checked: 85 instructions; 65 register moves, 21 stores, 21 loads, 0 slot moves, 9 rematerialized; 13 slots (92 bytes)
    x86_64 sdpa/0:
      reference checked: 157 instructions; 0 register moves, 213 stores, 253 loads, 0 slot moves, 0 rematerialized; 173 slots (1232 bytes)
      linear scan checked: 157 instructions; 81 register moves, 59 stores, 54 loads, 0 slot moves, 16 rematerialized; 22 slots (160 bytes)
    x86_64 conv/0:
      reference checked: 100 instructions; 0 register moves, 134 stores, 155 loads, 0 slot moves, 0 rematerialized; 109 slots (688 bytes)
      linear scan checked: 100 instructions; 59 register moves, 21 stores, 21 loads, 0 slot moves, 4 rematerialized; 17 slots (112 bytes) |}]

let%expect_test "linear-scan mutations with full pools" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: w1 does not hold %32
    aarch64 hole: NOT CAUGHT
    aarch64 live tie: NOT CAUGHT
    aarch64 needed eviction: NOT CAUGHT
    aarch64 split move: caught on 2 kernels; first softmax/0: checker: fn0 bb12: x19 does not hold %31
    x86_64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb10: xmm1.d does not hold %37
    x86_64 hole: caught on 3 kernels; first chain/0: checker: fn0 bb10: r14 does not hold %227
    x86_64 live tie: caught on 7 kernels; first chain/0: checker: fn0 bb4: r13d does not hold %41
    x86_64 needed eviction: caught on 4 kernels; first chain/0: physical verifier: allocated fn0 bb12 i278: target constraint: an instruction operand in memory
    x86_64 split move: caught on 5 kernels; first chain/0: checker: fn0 bb6: r9 does not hold %193 |}]
