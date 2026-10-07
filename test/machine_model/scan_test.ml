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

(* The hole mutation is caught on AArch64 under pressure and on x86-64 only
   with its full pools (one kernel): elsewhere the allocations it changes still
   pass the checker, so the intersections it ignores there are ones no read
   crosses. A needed eviction is caught under pressure only: with full pools
   no eviction meets an operand of the instruction that evicts. *)
let%expect_test "under pressure: both checked; what each mutation breaks" =
  A64_small.compare ();
  X64_small.compare ();
  A64_small.mutations ();
  X64_small.mutations ();
  [%expect
    {|
    aarch64, 3 registers chain/0:
      reference checked: 173 instructions; 0 register moves, 176 stores, 213 loads, 0 slot moves, 0 rematerialized; 167 slots (1112 bytes)
      linear scan checked: 173 instructions; 11 register moves, 33 stores, 28 loads, 0 slot moves, 13 rematerialized; 14 slots (68 bytes)
    aarch64, 3 registers chain/1:
      reference checked: 110 instructions; 0 register moves, 112 stores, 126 loads, 0 slot moves, 0 rematerialized; 109 slots (760 bytes)
      linear scan checked: 110 instructions; 10 register moves, 11 stores, 10 loads, 0 slot moves, 6 rematerialized; 3 slots (16 bytes)
    aarch64, 3 registers chain/2:
      reference checked: 64 instructions; 0 register moves, 67 stores, 78 loads, 0 slot moves, 0 rematerialized; 63 slots (420 bytes)
      linear scan checked: 64 instructions; 5 register moves, 8 stores, 8 loads, 0 slot moves, 5 rematerialized; 2 slots (8 bytes)
    aarch64, 3 registers bmm/0:
      reference checked: 92 instructions; 0 register moves, 97 stores, 109 loads, 0 slot moves, 0 rematerialized; 92 slots (620 bytes)
      linear scan checked: 92 instructions; 6 register moves, 13 stores, 13 loads, 0 slot moves, 9 rematerialized; 5 slots (24 bytes)
    aarch64, 3 registers softmax/0:
      reference checked: 304 instructions; 0 register moves, 259 stores, 333 loads, 0 slot moves, 0 rematerialized; 251 slots (1772 bytes)
      linear scan checked: 304 instructions; 9 register moves, 34 stores, 32 loads, 0 slot moves, 21 rematerialized; 9 slots (56 bytes)
    aarch64, 3 registers sdpa/0:
      reference checked: 715 instructions; 0 register moves, 558 stores, 768 loads, 0 slot moves, 0 rematerialized; 543 slots (3860 bytes)
      linear scan checked: 715 instructions; 22 register moves, 76 stores, 70 loads, 0 slot moves, 42 rematerialized; 13 slots (84 bytes)
    x86_64, 3 registers chain/0:
      reference checked: 175 instructions; 0 register moves, 172 stores, 213 loads, 0 slot moves, 0 rematerialized; 163 slots (1080 bytes)
      linear scan checked: 175 instructions; 58 register moves, 39 stores, 34 loads, 0 slot moves, 13 rematerialized; 16 slots (80 bytes)
    x86_64, 3 registers chain/1:
      reference checked: 104 instructions; 0 register moves, 103 stores, 123 loads, 0 slot moves, 0 rematerialized; 100 slots (688 bytes)
      linear scan checked: 104 instructions; 33 register moves, 25 stores, 23 loads, 0 slot moves, 5 rematerialized; 6 slots (36 bytes)
    x86_64, 3 registers chain/2:
      reference checked: 66 instructions; 0 register moves, 65 stores, 78 loads, 0 slot moves, 0 rematerialized; 61 slots (404 bytes)
      linear scan checked: 66 instructions; 20 register moves, 12 stores, 12 loads, 0 slot moves, 5 rematerialized; 4 slots (20 bytes)
    x86_64, 3 registers bmm/0:
      reference checked: 93 instructions; 0 register moves, 94 stores, 109 loads, 0 slot moves, 0 rematerialized; 89 slots (596 bytes)
      linear scan checked: 93 instructions; 30 register moves, 18 stores, 18 loads, 0 slot moves, 9 rematerialized; 6 slots (32 bytes)
    x86_64, 3 registers softmax/0:
      reference checked: 314 instructions; 0 register moves, 260 stores, 353 loads, 0 slot moves, 0 rematerialized; 252 slots (1780 bytes)
      linear scan checked: 314 instructions; 81 register moves, 44 stores, 44 loads, 0 slot moves, 27 rematerialized; 9 slots (56 bytes)
    x86_64, 3 registers sdpa/0:
      reference checked: 723 instructions; 0 register moves, 545 stores, 785 loads, 0 slot moves, 0 rematerialized; 530 slots (3756 bytes)
      linear scan checked: 723 instructions; 137 register moves, 86 stores, 86 loads, 0 slot moves, 52 rematerialized; 11 slots (68 bytes)
    aarch64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb12: d2 does not hold %36
    aarch64, 3 registers hole: caught on 4 kernels; first chain/0: checker: fn0 bb9: [slot3:4] does not hold %6
    aarch64, 3 registers needed eviction: caught on 6 kernels; first chain/0: physical verifier: allocated fn0 bb4: target constraint: an undeclared slot
    aarch64, 3 registers split move: caught on 6 kernels; first chain/0: checker: fn0 bb4: w2 does not hold %34
    x86_64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb24: xmm2.d does not hold %20
    x86_64, 3 registers hole: NOT CAUGHT
    x86_64, 3 registers needed eviction: caught on 5 kernels; first chain/0: physical verifier: allocated fn0 bb9 i218: target constraint: an instruction operand in memory
    x86_64, 3 registers split move: caught on 6 kernels; first chain/0: checker: fn0 bb4: esi does not hold %34 |}]

let%expect_test "both allocators pass the same checker; what each costs" =
  A64.compare ();
  X64.compare ();
  [%expect
    {|
    aarch64 chain/0:
      reference checked: 173 instructions; 0 register moves, 176 stores, 213 loads, 0 slot moves, 0 rematerialized; 167 slots (1112 bytes)
      linear scan checked: 173 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1:
      reference checked: 110 instructions; 0 register moves, 112 stores, 126 loads, 0 slot moves, 0 rematerialized; 109 slots (760 bytes)
      linear scan checked: 110 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2:
      reference checked: 64 instructions; 0 register moves, 67 stores, 78 loads, 0 slot moves, 0 rematerialized; 63 slots (420 bytes)
      linear scan checked: 64 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0:
      reference checked: 92 instructions; 0 register moves, 97 stores, 109 loads, 0 slot moves, 0 rematerialized; 92 slots (620 bytes)
      linear scan checked: 92 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0:
      reference checked: 304 instructions; 0 register moves, 259 stores, 333 loads, 0 slot moves, 0 rematerialized; 251 slots (1772 bytes)
      linear scan checked: 304 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    aarch64 sdpa/0:
      reference checked: 715 instructions; 0 register moves, 558 stores, 768 loads, 0 slot moves, 0 rematerialized; 543 slots (3860 bytes)
      linear scan checked: 715 instructions; 56 register moves, 2 stores, 1 loads, 0 slot moves, 7 rematerialized; 1 slots (8 bytes)
    x86_64 chain/0:
      reference checked: 175 instructions; 0 register moves, 172 stores, 213 loads, 0 slot moves, 0 rematerialized; 163 slots (1080 bytes)
      linear scan checked: 175 instructions; 64 register moves, 8 stores, 7 loads, 0 slot moves, 4 rematerialized; 5 slots (20 bytes)
    x86_64 chain/1:
      reference checked: 104 instructions; 0 register moves, 103 stores, 123 loads, 0 slot moves, 0 rematerialized; 100 slots (688 bytes)
      linear scan checked: 104 instructions; 35 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 chain/2:
      reference checked: 66 instructions; 0 register moves, 65 stores, 78 loads, 0 slot moves, 0 rematerialized; 61 slots (404 bytes)
      linear scan checked: 66 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0:
      reference checked: 93 instructions; 0 register moves, 94 stores, 109 loads, 0 slot moves, 0 rematerialized; 89 slots (596 bytes)
      linear scan checked: 93 instructions; 35 register moves, 0 stores, 0 loads, 0 slot moves, 3 rematerialized; 0 slots (0 bytes)
    x86_64 softmax/0:
      reference checked: 314 instructions; 0 register moves, 260 stores, 353 loads, 0 slot moves, 0 rematerialized; 252 slots (1780 bytes)
      linear scan checked: 314 instructions; 106 register moves, 10 stores, 10 loads, 0 slot moves, 12 rematerialized; 5 slots (32 bytes)
    x86_64 sdpa/0:
      reference checked: 723 instructions; 0 register moves, 545 stores, 785 loads, 0 slot moves, 0 rematerialized; 530 slots (3756 bytes)
      linear scan checked: 723 instructions; 159 register moves, 30 stores, 30 loads, 0 slot moves, 26 rematerialized; 7 slots (44 bytes) |}]

let%expect_test "linear-scan mutations with full pools" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb2: w1 does not hold %31
    aarch64 hole: NOT CAUGHT
    aarch64 needed eviction: NOT CAUGHT
    aarch64 split move: caught on 2 kernels; first softmax/0: checker: fn0 bb22: w20 does not hold %24
    x86_64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb4: eax does not hold %30
    x86_64 hole: caught on 1 kernels; first sdpa/0: checker: fn0 bb29: r13d does not hold %22
    x86_64 needed eviction: NOT CAUGHT
    x86_64 split move: caught on 3 kernels; first chain/0: checker: fn0 bb9: ebx does not hold %6 |}]
