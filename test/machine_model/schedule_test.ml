(* M10.2: block-local scheduling of the model kernels' selected programs on
   both targets — every schedule within its source order's dependences and
   reverified, what it moves, what linear scan then costs, and each dropped
   dependence class caught. *)

open Machine_ir

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
  module Sch = Machine_alloc.Mir_schedule.Make (T)
  module Ls = Machine_alloc.Mir_linear_scan.Make (T) (R)
  module Policy = Machine_alloc.Mir_schedule.Policy

  let moved (before : Sch.S.Verified.t) (after : Sch.S.Verified.t) =
    let bodies v =
      List.concat_map
        (fun (f : (_, _) Mir_func.t) ->
          List.map
            (fun (b : (_, _) Mir_block.t) ->
              List.map
                (fun (i : _ Mir_instr.t) -> i.Mir_instr.id)
                b.Mir_block.body)
            f.Mir_func.blocks)
        (Sch.S.Verified.selected v).Sch.S.program.Mir_program.funcs
    in
    List.fold_left2
      (fun n a b ->
        List.fold_left2 (fun n x y -> if x <> y then n + 1 else n) n a b)
      0 (bodies before) (bodies after)

  let compare () =
    List.iter
      (fun (name, gen, sites) ->
        let v = X.select ~sites gen in
        match (Sch.schedule Policy.Sink v, Sch.schedule Policy.Reverse v) with
        | Error r, _ | _, Error r ->
            Fmt.pr "%s %s: %a@." X.name name
              Machine_alloc.Mir_schedule.Refusal.pp r
        | Ok s, Ok r ->
            Fmt.pr
              "%s %s: sink moves %d (%a), reverse %d@.  source: %a@.  sink: \
               %a@."
              X.name name (moved v s) Mir_id.Revision.pp
              (Sch.S.Verified.selected s).Sch.S.program.Mir_program.revision
              (moved v r) Machine_alloc.Mir_alloc_stats.pp
              (Machine_alloc.Mir_alloc_stats.of_program (Ls.allocate v))
              Machine_alloc.Mir_alloc_stats.pp
              (Machine_alloc.Mir_alloc_stats.of_program (Ls.allocate s)))
      (Scan_test.kernels ())

  (* Each class dropped from the graph under the reverse order: the check
     refuses, and how often the selected verifier alone rejects the same order.
     The graph keeps a flags writer on the side of a condition range it started
     on, which is stricter than the verifier's rule: on AArch64 every writer the
     reverse order moves lands outside any range, so only the check sees it. *)
  let mutations () =
    List.iter
      (fun (label, mutation) ->
        let caught =
          List.filter_map
            (fun (name, gen, sites) ->
              let v = X.select ~sites gen in
              match Sch.schedule ~mutation Policy.Reverse v with
              | Ok _ -> None
              | Error r ->
                  let verifier =
                    Result.is_error
                      (Err.payload
                         (Sch.S.Verified.verify
                            (Sch.reorder ~mutation Policy.Reverse v)))
                  in
                  Some (name, r, verifier))
            (Scan_test.kernels ())
        in
        match caught with
        | [] -> Fmt.pr "%s %s: NOT CAUGHT@." X.name label
        | (name, r, _) :: _ ->
            Fmt.pr
              "%s %s: caught on %d kernels (verifier alone: %d); first %s: %a@."
              X.name label (List.length caught)
              (List.length (List.filter (fun (_, _, v) -> v) caught))
              name Machine_alloc.Mir_schedule.Refusal.pp r)
      Machine_alloc.Mir_schedule.Mutation.
        [ ("data", Data); ("flags", Flags); ("order", Order) ]
end

module A64 =
  Run (Machine_target_aarch64.A64) (Machine_target_aarch64.A64_regs)
    (struct
      include Scan_test.A64_select_

      let name = "aarch64"
    end)

module X64 =
  Run (Machine_target_x86_64.X64) (Machine_target_x86_64.X64_regs)
    (struct
      include Scan_test.X64_select_

      let name = "x86_64"
    end)

module A64_small =
  Run
    (Machine_target_aarch64.A64)
    (struct
      include Machine_target_aarch64.A64_regs

      let allocatable bank = Scan_test.take 3 (allocatable bank)
    end)
    (struct
      include Scan_test.A64_select_

      let name = "aarch64, 3 registers"
    end)

module X64_small =
  Run
    (Machine_target_x86_64.X64)
    (struct
      include Machine_target_x86_64.X64_regs

      let allocatable bank = Scan_test.take 3 (allocatable bank)
    end)
    (struct
      include Scan_test.X64_select_

      let name = "x86_64, 3 registers"
    end)

let%expect_test "dropped dependences are caught" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 data: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb1: i157 scheduled after i158, which depends on it
    aarch64 flags: caught on 3 kernels (verifier alone: 0); first chain/0: dependence: bb4: i167 scheduled after i179, which depends on it
    aarch64 order: caught on 5 kernels (verifier alone: 5); first chain/0: dependence: bb19: i276 scheduled after i295, which depends on it
    x86_64 data: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb1: i157 scheduled after i158, which depends on it
    x86_64 flags: caught on 4 kernels (verifier alone: 3); first chain/0: dependence: bb4: i168 scheduled after i180, which depends on it
    x86_64 order: caught on 5 kernels (verifier alone: 5); first chain/0: dependence: bb19: i280 scheduled after i298, which depends on it |}]

let%expect_test "sink scheduling, then linear scan" =
  A64.compare ();
  X64.compare ();
  A64_small.compare ();
  X64_small.compare ();
  [%expect
    {|
    aarch64 chain/0: sink moves 4 (rev1), reverse 141
      source: 173 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 173 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1: sink moves 0 (rev0), reverse 99
      source: 110 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 110 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2: sink moves 0 (rev0), reverse 47
      source: 64 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 64 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0: sink moves 0 (rev0), reverse 78
      source: 92 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 92 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0: sink moves 0 (rev0), reverse 273
      source: 304 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
      sink: 304 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 1 rematerialized; 0 slots (0 bytes)
    aarch64 sdpa/0: sink moves 0 (rev0), reverse 655
      source: 715 instructions; 56 register moves, 2 stores, 1 loads, 0 slot moves, 7 rematerialized; 1 slots (8 bytes)
      sink: 715 instructions; 56 register moves, 2 stores, 1 loads, 0 slot moves, 7 rematerialized; 1 slots (8 bytes)
    x86_64 chain/0: sink moves 4 (rev1), reverse 131
      source: 175 instructions; 64 register moves, 8 stores, 7 loads, 0 slot moves, 4 rematerialized; 5 slots (20 bytes)
      sink: 175 instructions; 64 register moves, 8 stores, 7 loads, 0 slot moves, 4 rematerialized; 5 slots (20 bytes)
    x86_64 chain/1: sink moves 0 (rev0), reverse 87
      source: 104 instructions; 35 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 104 instructions; 35 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 chain/2: sink moves 0 (rev0), reverse 49
      source: 66 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 66 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0: sink moves 0 (rev0), reverse 75
      source: 93 instructions; 35 register moves, 0 stores, 0 loads, 0 slot moves, 3 rematerialized; 0 slots (0 bytes)
      sink: 93 instructions; 35 register moves, 0 stores, 0 loads, 0 slot moves, 3 rematerialized; 0 slots (0 bytes)
    x86_64 softmax/0: sink moves 4 (rev1), reverse 277
      source: 314 instructions; 106 register moves, 10 stores, 10 loads, 0 slot moves, 12 rematerialized; 5 slots (32 bytes)
      sink: 314 instructions; 106 register moves, 10 stores, 10 loads, 0 slot moves, 12 rematerialized; 5 slots (32 bytes)
    x86_64 sdpa/0: sink moves 4 (rev1), reverse 640
      source: 723 instructions; 159 register moves, 30 stores, 30 loads, 0 slot moves, 26 rematerialized; 7 slots (44 bytes)
      sink: 723 instructions; 159 register moves, 30 stores, 30 loads, 0 slot moves, 26 rematerialized; 7 slots (44 bytes)
    aarch64, 3 registers chain/0: sink moves 4 (rev1), reverse 141
      source: 173 instructions; 11 register moves, 33 stores, 28 loads, 0 slot moves, 13 rematerialized; 14 slots (68 bytes)
      sink: 173 instructions; 11 register moves, 33 stores, 28 loads, 0 slot moves, 13 rematerialized; 14 slots (68 bytes)
    aarch64, 3 registers chain/1: sink moves 0 (rev0), reverse 99
      source: 110 instructions; 10 register moves, 11 stores, 10 loads, 0 slot moves, 6 rematerialized; 3 slots (16 bytes)
      sink: 110 instructions; 10 register moves, 11 stores, 10 loads, 0 slot moves, 6 rematerialized; 3 slots (16 bytes)
    aarch64, 3 registers chain/2: sink moves 0 (rev0), reverse 47
      source: 64 instructions; 5 register moves, 8 stores, 8 loads, 0 slot moves, 5 rematerialized; 2 slots (8 bytes)
      sink: 64 instructions; 5 register moves, 8 stores, 8 loads, 0 slot moves, 5 rematerialized; 2 slots (8 bytes)
    aarch64, 3 registers bmm/0: sink moves 0 (rev0), reverse 78
      source: 92 instructions; 6 register moves, 13 stores, 13 loads, 0 slot moves, 9 rematerialized; 5 slots (24 bytes)
      sink: 92 instructions; 6 register moves, 13 stores, 13 loads, 0 slot moves, 9 rematerialized; 5 slots (24 bytes)
    aarch64, 3 registers softmax/0: sink moves 0 (rev0), reverse 273
      source: 304 instructions; 9 register moves, 34 stores, 32 loads, 0 slot moves, 21 rematerialized; 9 slots (56 bytes)
      sink: 304 instructions; 9 register moves, 34 stores, 32 loads, 0 slot moves, 21 rematerialized; 9 slots (56 bytes)
    aarch64, 3 registers sdpa/0: sink moves 0 (rev0), reverse 655
      source: 715 instructions; 22 register moves, 76 stores, 70 loads, 0 slot moves, 42 rematerialized; 13 slots (84 bytes)
      sink: 715 instructions; 22 register moves, 76 stores, 70 loads, 0 slot moves, 42 rematerialized; 13 slots (84 bytes)
    x86_64, 3 registers chain/0: sink moves 4 (rev1), reverse 131
      source: 175 instructions; 58 register moves, 39 stores, 34 loads, 0 slot moves, 13 rematerialized; 16 slots (80 bytes)
      sink: 175 instructions; 58 register moves, 39 stores, 34 loads, 0 slot moves, 13 rematerialized; 16 slots (80 bytes)
    x86_64, 3 registers chain/1: sink moves 0 (rev0), reverse 87
      source: 104 instructions; 33 register moves, 25 stores, 23 loads, 0 slot moves, 5 rematerialized; 6 slots (36 bytes)
      sink: 104 instructions; 33 register moves, 25 stores, 23 loads, 0 slot moves, 5 rematerialized; 6 slots (36 bytes)
    x86_64, 3 registers chain/2: sink moves 0 (rev0), reverse 49
      source: 66 instructions; 20 register moves, 12 stores, 12 loads, 0 slot moves, 5 rematerialized; 4 slots (20 bytes)
      sink: 66 instructions; 20 register moves, 12 stores, 12 loads, 0 slot moves, 5 rematerialized; 4 slots (20 bytes)
    x86_64, 3 registers bmm/0: sink moves 0 (rev0), reverse 75
      source: 93 instructions; 30 register moves, 18 stores, 18 loads, 0 slot moves, 9 rematerialized; 6 slots (32 bytes)
      sink: 93 instructions; 30 register moves, 18 stores, 18 loads, 0 slot moves, 9 rematerialized; 6 slots (32 bytes)
    x86_64, 3 registers softmax/0: sink moves 4 (rev1), reverse 277
      source: 314 instructions; 81 register moves, 44 stores, 44 loads, 0 slot moves, 27 rematerialized; 9 slots (56 bytes)
      sink: 314 instructions; 82 register moves, 46 stores, 46 loads, 0 slot moves, 27 rematerialized; 9 slots (56 bytes)
    x86_64, 3 registers sdpa/0: sink moves 4 (rev1), reverse 640
      source: 723 instructions; 137 register moves, 86 stores, 86 loads, 0 slot moves, 52 rematerialized; 11 slots (68 bytes)
      sink: 723 instructions; 136 register moves, 88 stores, 88 loads, 0 slot moves, 52 rematerialized; 11 slots (68 bytes) |}]
