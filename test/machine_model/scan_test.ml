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

(* The hole mutation is caught only on AArch64 under pressure. On x86-64 it
   changes the allocation of four kernels, and each changed one still passes the
   checker, so the intersections it ignores there are ones no read crosses.
   With full pools no hole decides a register. *)
let%expect_test "under pressure: both checked; what each mutation breaks" =
  A64_small.compare ();
  X64_small.compare ();
  A64_small.mutations ();
  X64_small.mutations ();
  [%expect
    {|
    aarch64, 3 registers chain/0:
      reference checked: 225 instructions; 0 register moves, 228 stores, 278 loads, 0 slot moves, 0 rematerialized; 219 slots (1528 bytes)
      linear scan checked: 225 instructions; 11 register moves, 33 stores, 33 loads, 0 slot moves, 25 rematerialized; 15 slots (72 bytes)
    aarch64, 3 registers chain/1:
      reference checked: 214 instructions; 0 register moves, 216 stores, 256 loads, 0 slot moves, 0 rematerialized; 213 slots (1592 bytes)
      linear scan checked: 214 instructions; 8 register moves, 16 stores, 16 loads, 0 slot moves, 26 rematerialized; 4 slots (20 bytes)
    aarch64, 3 registers chain/2:
      reference checked: 88 instructions; 0 register moves, 91 stores, 108 loads, 0 slot moves, 0 rematerialized; 87 slots (612 bytes)
      linear scan checked: 88 instructions; 6 register moves, 11 stores, 11 loads, 0 slot moves, 9 rematerialized; 3 slots (12 bytes)
    aarch64, 3 registers bmm/0:
      reference checked: 128 instructions; 0 register moves, 133 stores, 154 loads, 0 slot moves, 0 rematerialized; 128 slots (908 bytes)
      linear scan checked: 128 instructions; 7 register moves, 15 stores, 17 loads, 0 slot moves, 17 rematerialized; 5 slots (24 bytes)
    aarch64, 3 registers softmax/0:
      reference checked: 336 instructions; 0 register moves, 291 stores, 373 loads, 0 slot moves, 0 rematerialized; 283 slots (2028 bytes)
      linear scan checked: 336 instructions; 11 register moves, 39 stores, 39 loads, 0 slot moves, 28 rematerialized; 9 slots (56 bytes)
    aarch64, 3 registers sdpa/0:
      reference checked: 755 instructions; 0 register moves, 598 stores, 818 loads, 0 slot moves, 0 rematerialized; 583 slots (4180 bytes)
      linear scan checked: 755 instructions; 22 register moves, 80 stores, 79 loads, 0 slot moves, 53 rematerialized; 13 slots (84 bytes)
    x86_64, 3 registers chain/0:
      reference checked: 227 instructions; 0 register moves, 224 stores, 278 loads, 0 slot moves, 0 rematerialized; 215 slots (1496 bytes)
      linear scan checked: 227 instructions; 85 register moves, 59 stores, 56 loads, 0 slot moves, 28 rematerialized; 16 slots (80 bytes)
    x86_64, 3 registers chain/1:
      reference checked: 208 instructions; 0 register moves, 207 stores, 253 loads, 0 slot moves, 0 rematerialized; 204 slots (1520 bytes)
      linear scan checked: 208 instructions; 85 register moves, 54 stores, 48 loads, 0 slot moves, 27 rematerialized; 6 slots (36 bytes)
    x86_64, 3 registers chain/2:
      reference checked: 90 instructions; 0 register moves, 89 stores, 108 loads, 0 slot moves, 0 rematerialized; 85 slots (596 bytes)
      linear scan checked: 90 instructions; 32 register moves, 22 stores, 20 loads, 0 slot moves, 11 rematerialized; 4 slots (20 bytes)
    x86_64, 3 registers bmm/0:
      reference checked: 129 instructions; 0 register moves, 130 stores, 154 loads, 0 slot moves, 0 rematerialized; 125 slots (884 bytes)
      linear scan checked: 129 instructions; 48 register moves, 33 stores, 31 loads, 0 slot moves, 18 rematerialized; 6 slots (32 bytes)
    x86_64, 3 registers softmax/0:
      reference checked: 346 instructions; 0 register moves, 292 stores, 393 loads, 0 slot moves, 0 rematerialized; 284 slots (2036 bytes)
      linear scan checked: 346 instructions; 99 register moves, 72 stores, 63 loads, 0 slot moves, 43 rematerialized; 9 slots (56 bytes)
    x86_64, 3 registers sdpa/0:
      reference checked: 763 instructions; 0 register moves, 585 stores, 835 loads, 0 slot moves, 0 rematerialized; 570 slots (4076 bytes)
      linear scan checked: 763 instructions; 158 register moves, 120 stores, 102 loads, 0 slot moves, 74 rematerialized; 11 slots (68 bytes)
    aarch64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb12: d2 does not hold %36
    aarch64, 3 registers hole: caught on 2 kernels; first bmm/0: checker: fn0 bb13: w2 does not hold %14
    aarch64, 3 registers split move: caught on 6 kernels; first chain/0: checker: fn0 bb4: w2 does not hold %34
    x86_64, 3 registers call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb24: xmm2.d does not hold %20
    x86_64, 3 registers hole: NOT CAUGHT
    x86_64, 3 registers split move: caught on 6 kernels; first chain/0: checker: fn0 bb4: ebx does not hold %34 |}]

let%expect_test "both allocators pass the same checker; what each costs" =
  A64.compare ();
  X64.compare ();
  [%expect
    {|
    aarch64 chain/0:
      reference checked: 225 instructions; 0 register moves, 228 stores, 278 loads, 0 slot moves, 0 rematerialized; 219 slots (1528 bytes)
      linear scan checked: 225 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1:
      reference checked: 214 instructions; 0 register moves, 216 stores, 256 loads, 0 slot moves, 0 rematerialized; 213 slots (1592 bytes)
      linear scan checked: 214 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2:
      reference checked: 88 instructions; 0 register moves, 91 stores, 108 loads, 0 slot moves, 0 rematerialized; 87 slots (612 bytes)
      linear scan checked: 88 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0:
      reference checked: 128 instructions; 0 register moves, 133 stores, 154 loads, 0 slot moves, 0 rematerialized; 128 slots (908 bytes)
      linear scan checked: 128 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0:
      reference checked: 336 instructions; 0 register moves, 291 stores, 373 loads, 0 slot moves, 0 rematerialized; 283 slots (2028 bytes)
      linear scan checked: 336 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 3 rematerialized; 0 slots (0 bytes)
    aarch64 sdpa/0:
      reference checked: 755 instructions; 0 register moves, 598 stores, 818 loads, 0 slot moves, 0 rematerialized; 583 slots (4180 bytes)
      linear scan checked: 755 instructions; 56 register moves, 2 stores, 2 loads, 0 slot moves, 16 rematerialized; 1 slots (8 bytes)
    x86_64 chain/0:
      reference checked: 227 instructions; 0 register moves, 224 stores, 278 loads, 0 slot moves, 0 rematerialized; 215 slots (1496 bytes)
      linear scan checked: 227 instructions; 89 register moves, 21 stores, 20 loads, 0 slot moves, 15 rematerialized; 11 slots (44 bytes)
    x86_64 chain/1:
      reference checked: 208 instructions; 0 register moves, 207 stores, 253 loads, 0 slot moves, 0 rematerialized; 204 slots (1520 bytes)
      linear scan checked: 208 instructions; 88 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 2 slots (8 bytes)
    x86_64 chain/2:
      reference checked: 90 instructions; 0 register moves, 89 stores, 108 loads, 0 slot moves, 0 rematerialized; 85 slots (596 bytes)
      linear scan checked: 90 instructions; 35 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 2 slots (8 bytes)
    x86_64 bmm/0:
      reference checked: 129 instructions; 0 register moves, 130 stores, 154 loads, 0 slot moves, 0 rematerialized; 125 slots (884 bytes)
      linear scan checked: 129 instructions; 53 register moves, 10 stores, 10 loads, 0 slot moves, 8 rematerialized; 3 slots (12 bytes)
    x86_64 softmax/0:
      reference checked: 346 instructions; 0 register moves, 292 stores, 393 loads, 0 slot moves, 0 rematerialized; 284 slots (2036 bytes)
      linear scan checked: 346 instructions; 103 register moves, 30 stores, 27 loads, 0 slot moves, 26 rematerialized; 7 slots (40 bytes)
    x86_64 sdpa/0:
      reference checked: 763 instructions; 0 register moves, 585 stores, 835 loads, 0 slot moves, 0 rematerialized; 570 slots (4076 bytes)
      linear scan checked: 763 instructions; 172 register moves, 54 stores, 46 loads, 0 slot moves, 59 rematerialized; 8 slots (48 bytes) |}]

let%expect_test "linear-scan mutations with full pools" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb2: w1 does not hold %31
    aarch64 hole: NOT CAUGHT
    aarch64 split move: caught on 2 kernels; first softmax/0: checker: fn0 bb22: w20 does not hold %24
    x86_64 call interval: caught on 2 kernels; first softmax/0: checker: fn0 bb10: xmm1.d does not hold %35
    x86_64 hole: NOT CAUGHT
    x86_64 split move: caught on 6 kernels; first chain/0: checker: fn0 bb4: r13d does not hold %35 |}]
