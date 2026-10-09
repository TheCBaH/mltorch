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
    aarch64 data: caught on 7 kernels (verifier alone: 7); first chain/0: dependence: bb1: i210 scheduled after i211, which depends on it
    aarch64 flags: caught on 4 kernels (verifier alone: 0); first chain/0: dependence: bb4: i219 scheduled after i225, which depends on it
    aarch64 order: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb19: i322 scheduled after i327, which depends on it
    x86_64 data: caught on 7 kernels (verifier alone: 7); first chain/0: dependence: bb1: i210 scheduled after i211, which depends on it
    x86_64 flags: caught on 4 kernels (verifier alone: 2); first chain/0: dependence: bb4: i219 scheduled after i225, which depends on it
    x86_64 order: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb19: i328 scheduled after i333, which depends on it |}]

let%expect_test "sink scheduling, then linear scan" =
  A64.compare ();
  X64.compare ();
  A64_small.compare ();
  X64_small.compare ();
  [%expect
    {|
    aarch64 chain/0: sink moves 8 (rev1), reverse 81
      source: 99 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 99 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1: sink moves 0 (rev0), reverse 50
      source: 57 instructions; 6 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 57 instructions; 6 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2: sink moves 0 (rev0), reverse 20
      source: 30 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 30 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0: sink moves 3 (rev1), reverse 44
      source: 55 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 55 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0: sink moves 4 (rev1), reverse 190
      source: 211 instructions; 18 register moves, 6 stores, 6 loads, 0 slot moves, 0 rematerialized; 2 slots (8 bytes)
      sink: 211 instructions; 17 register moves, 3 stores, 3 loads, 0 slot moves, 0 rematerialized; 1 slots (4 bytes)
    aarch64 sdpa/0: sink moves 11 (rev1), reverse 489
      source: 549 instructions; 31 register moves, 7 stores, 6 loads, 0 slot moves, 0 rematerialized; 2 slots (12 bytes)
      sink: 549 instructions; 30 register moves, 3 stores, 2 loads, 0 slot moves, 0 rematerialized; 1 slots (8 bytes)
    aarch64 conv/0: sink moves 8 (rev1), reverse 100
      source: 119 instructions; 19 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 119 instructions; 19 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 chain/0: sink moves 38 (rev1), reverse 74
      source: 98 instructions; 26 register moves, 4 stores, 4 loads, 0 slot moves, 2 rematerialized; 4 slots (20 bytes)
      sink: 98 instructions; 25 register moves, 4 stores, 4 loads, 0 slot moves, 2 rematerialized; 4 slots (20 bytes)
    x86_64 chain/1: sink moves 15 (rev1), reverse 57
      source: 63 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 63 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 chain/2: sink moves 15 (rev1), reverse 24
      source: 35 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 35 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 bmm/0: sink moves 21 (rev1), reverse 44
      source: 52 instructions; 13 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 52 instructions; 13 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 softmax/0: sink moves 47 (rev1), reverse 213
      source: 232 instructions; 38 register moves, 21 stores, 20 loads, 0 slot moves, 3 rematerialized; 7 slots (44 bytes)
      sink: 232 instructions; 38 register moves, 21 stores, 20 loads, 0 slot moves, 3 rematerialized; 7 slots (44 bytes)
    x86_64 sdpa/0: sink moves 60 (rev1), reverse 528
      source: 573 instructions; 55 register moves, 38 stores, 35 loads, 0 slot moves, 13 rematerialized; 8 slots (52 bytes)
      sink: 573 instructions; 52 register moves, 38 stores, 35 loads, 0 slot moves, 13 rematerialized; 8 slots (52 bytes)
    x86_64 conv/0: sink moves 44 (rev1), reverse 90
      source: 114 instructions; 30 register moves, 12 stores, 12 loads, 0 slot moves, 2 rematerialized; 6 slots (28 bytes)
      sink: 114 instructions; 30 register moves, 9 stores, 9 loads, 0 slot moves, 2 rematerialized; 6 slots (28 bytes)
    aarch64, 3 registers chain/0: sink moves 8 (rev1), reverse 81
      source: 99 instructions; 12 register moves, 47 stores, 38 loads, 0 slot moves, 5 rematerialized; 15 slots (80 bytes)
      sink: 99 instructions; 11 register moves, 45 stores, 36 loads, 0 slot moves, 6 rematerialized; 15 slots (80 bytes)
    aarch64, 3 registers chain/1: sink moves 0 (rev0), reverse 50
      source: 57 instructions; 9 register moves, 9 stores, 7 loads, 0 slot moves, 2 rematerialized; 4 slots (24 bytes)
      sink: 57 instructions; 9 register moves, 9 stores, 7 loads, 0 slot moves, 2 rematerialized; 4 slots (24 bytes)
    aarch64, 3 registers chain/2: sink moves 0 (rev0), reverse 20
      source: 30 instructions; 7 register moves, 6 stores, 5 loads, 0 slot moves, 2 rematerialized; 3 slots (16 bytes)
      sink: 30 instructions; 7 register moves, 6 stores, 5 loads, 0 slot moves, 2 rematerialized; 3 slots (16 bytes)
    aarch64, 3 registers bmm/0: sink moves 3 (rev1), reverse 44
      source: 55 instructions; 12 register moves, 19 stores, 16 loads, 0 slot moves, 3 rematerialized; 7 slots (44 bytes)
      sink: 55 instructions; 12 register moves, 18 stores, 16 loads, 0 slot moves, 3 rematerialized; 7 slots (40 bytes)
    aarch64, 3 registers softmax/0: sink moves 4 (rev1), reverse 190
      source: 211 instructions; 19 register moves, 37 stores, 33 loads, 0 slot moves, 9 rematerialized; 13 slots (88 bytes)
      sink: 211 instructions; 19 register moves, 36 stores, 32 loads, 0 slot moves, 10 rematerialized; 12 slots (80 bytes)
    aarch64, 3 registers sdpa/0: sink moves 11 (rev1), reverse 489
      source: 549 instructions; 31 register moves, 87 stores, 75 loads, 0 slot moves, 16 rematerialized; 16 slots (108 bytes)
      sink: 549 instructions; 30 register moves, 86 stores, 74 loads, 0 slot moves, 17 rematerialized; 16 slots (108 bytes)
    aarch64, 3 registers conv/0: sink moves 8 (rev1), reverse 100
      source: 119 instructions; 12 register moves, 52 stores, 43 loads, 0 slot moves, 8 rematerialized; 16 slots (84 bytes)
      sink: 119 instructions; 12 register moves, 50 stores, 41 loads, 0 slot moves, 8 rematerialized; 16 slots (84 bytes)
    x86_64, 3 registers chain/0: sink moves 38 (rev1), reverse 74
      source: 98 instructions; 14 register moves, 47 stores, 44 loads, 0 slot moves, 8 rematerialized; 14 slots (76 bytes)
      sink: 98 instructions; 17 register moves, 45 stores, 41 loads, 0 slot moves, 5 rematerialized; 14 slots (76 bytes)
    x86_64, 3 registers chain/1: sink moves 15 (rev1), reverse 57
      source: 63 instructions; 9 register moves, 15 stores, 14 loads, 0 slot moves, 4 rematerialized; 7 slots (44 bytes)
      sink: 63 instructions; 10 register moves, 15 stores, 14 loads, 0 slot moves, 2 rematerialized; 7 slots (44 bytes)
    x86_64, 3 registers chain/2: sink moves 15 (rev1), reverse 24
      source: 35 instructions; 9 register moves, 11 stores, 10 loads, 0 slot moves, 4 rematerialized; 5 slots (28 bytes)
      sink: 35 instructions; 10 register moves, 11 stores, 10 loads, 0 slot moves, 2 rematerialized; 5 slots (28 bytes)
    x86_64, 3 registers bmm/0: sink moves 21 (rev1), reverse 44
      source: 52 instructions; 10 register moves, 17 stores, 17 loads, 0 slot moves, 6 rematerialized; 8 slots (48 bytes)
      sink: 52 instructions; 11 register moves, 17 stores, 17 loads, 0 slot moves, 3 rematerialized; 8 slots (48 bytes)
    x86_64, 3 registers softmax/0: sink moves 47 (rev1), reverse 213
      source: 232 instructions; 23 register moves, 37 stores, 39 loads, 0 slot moves, 21 rematerialized; 10 slots (64 bytes)
      sink: 232 instructions; 28 register moves, 39 stores, 42 loads, 0 slot moves, 17 rematerialized; 10 slots (64 bytes)
    x86_64, 3 registers sdpa/0: sink moves 60 (rev1), reverse 528
      source: 573 instructions; 38 register moves, 70 stores, 71 loads, 0 slot moves, 43 rematerialized; 12 slots (76 bytes)
      sink: 573 instructions; 45 register moves, 73 stores, 74 loads, 0 slot moves, 38 rematerialized; 12 slots (76 bytes)
    x86_64, 3 registers conv/0: sink moves 44 (rev1), reverse 90
      source: 114 instructions; 15 register moves, 52 stores, 52 loads, 0 slot moves, 9 rematerialized; 16 slots (84 bytes)
      sink: 114 instructions; 17 register moves, 52 stores, 51 loads, 0 slot moves, 6 rematerialized; 16 slots (84 bytes) |}]
