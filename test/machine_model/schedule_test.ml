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
    aarch64 data: caught on 7 kernels (verifier alone: 7); first chain/0: dependence: bb1: i205 scheduled after i206, which depends on it
    aarch64 flags: caught on 2 kernels (verifier alone: 0); first chain/0: dependence: bb4: i222 scheduled after i228, which depends on it
    aarch64 order: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb19: i314 scheduled after i315, which depends on it
    x86_64 data: caught on 7 kernels (verifier alone: 7); first chain/0: dependence: bb1: i205 scheduled after i206, which depends on it
    x86_64 flags: caught on 3 kernels (verifier alone: 0); first chain/0: dependence: bb4: i218 scheduled after i224, which depends on it
    x86_64 order: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb19: i317 scheduled after i318, which depends on it |}]

let%expect_test "sink scheduling, then linear scan" =
  A64.compare ();
  X64.compare ();
  A64_small.compare ();
  X64_small.compare ();
  [%expect
    {|
    aarch64 chain/0: sink moves 8 (rev1), reverse 73
      source: 91 instructions; 39 register moves, 1 stores, 1 loads, 0 slot moves, 2 rematerialized; 1 slots (8 bytes)
      sink: 91 instructions; 39 register moves, 1 stores, 1 loads, 0 slot moves, 2 rematerialized; 1 slots (8 bytes)
    aarch64 chain/1: sink moves 0 (rev0), reverse 52
      source: 58 instructions; 21 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 58 instructions; 21 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2: sink moves 0 (rev0), reverse 24
      source: 30 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 30 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0: sink moves 0 (rev0), reverse 32
      source: 38 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 38 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0: sink moves 0 (rev0), reverse 46
      source: 64 instructions; 55 register moves, 5 stores, 5 loads, 0 slot moves, 6 rematerialized; 3 slots (24 bytes)
      sink: 64 instructions; 55 register moves, 5 stores, 5 loads, 0 slot moves, 6 rematerialized; 3 slots (24 bytes)
    aarch64 sdpa/0: sink moves 0 (rev0), reverse 101
      source: 127 instructions; 97 register moves, 36 stores, 35 loads, 0 slot moves, 0 rematerialized; 13 slots (92 bytes)
      sink: 127 instructions; 97 register moves, 36 stores, 35 loads, 0 slot moves, 0 rematerialized; 13 slots (92 bytes)
    aarch64 conv/0: sink moves 8 (rev1), reverse 84
      source: 105 instructions; 46 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 5 slots (40 bytes)
      sink: 105 instructions; 46 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 5 slots (40 bytes)
    x86_64 chain/0: sink moves 8 (rev1), reverse 63
      source: 92 instructions; 45 register moves, 18 stores, 18 loads, 0 slot moves, 4 rematerialized; 14 slots (92 bytes)
      sink: 92 instructions; 45 register moves, 18 stores, 18 loads, 0 slot moves, 4 rematerialized; 14 slots (92 bytes)
    x86_64 chain/1: sink moves 0 (rev0), reverse 48
      source: 55 instructions; 23 register moves, 5 stores, 5 loads, 0 slot moves, 5 rematerialized; 4 slots (28 bytes)
      sink: 55 instructions; 23 register moves, 5 stores, 5 loads, 0 slot moves, 5 rematerialized; 4 slots (28 bytes)
    x86_64 chain/2: sink moves 0 (rev0), reverse 30
      source: 36 instructions; 19 register moves, 1 stores, 1 loads, 0 slot moves, 1 rematerialized; 1 slots (8 bytes)
      sink: 36 instructions; 19 register moves, 1 stores, 1 loads, 0 slot moves, 1 rematerialized; 1 slots (8 bytes)
    x86_64 bmm/0: sink moves 0 (rev0), reverse 32
      source: 38 instructions; 24 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
      sink: 38 instructions; 24 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (28 bytes)
    x86_64 softmax/0: sink moves 8 (rev1), reverse 69
      source: 85 instructions; 65 register moves, 21 stores, 21 loads, 0 slot moves, 9 rematerialized; 13 slots (92 bytes)
      sink: 85 instructions; 65 register moves, 21 stores, 21 loads, 0 slot moves, 9 rematerialized; 13 slots (92 bytes)
    x86_64 sdpa/0: sink moves 8 (rev1), reverse 129
      source: 157 instructions; 81 register moves, 59 stores, 54 loads, 0 slot moves, 16 rematerialized; 22 slots (160 bytes)
      sink: 157 instructions; 81 register moves, 59 stores, 54 loads, 0 slot moves, 16 rematerialized; 22 slots (160 bytes)
    x86_64 conv/0: sink moves 8 (rev1), reverse 69
      source: 100 instructions; 59 register moves, 21 stores, 21 loads, 0 slot moves, 4 rematerialized; 17 slots (112 bytes)
      sink: 100 instructions; 59 register moves, 21 stores, 21 loads, 0 slot moves, 4 rematerialized; 17 slots (112 bytes)
    aarch64, 3 registers chain/0: sink moves 8 (rev1), reverse 73
      source: 91 instructions; 11 register moves, 62 stores, 51 loads, 0 slot moves, 6 rematerialized; 25 slots (164 bytes)
      sink: 91 instructions; 11 register moves, 62 stores, 51 loads, 0 slot moves, 6 rematerialized; 25 slots (164 bytes)
    aarch64, 3 registers chain/1: sink moves 0 (rev0), reverse 52
      source: 58 instructions; 4 register moves, 39 stores, 30 loads, 0 slot moves, 3 rematerialized; 17 slots (124 bytes)
      sink: 58 instructions; 4 register moves, 39 stores, 30 loads, 0 slot moves, 3 rematerialized; 17 slots (124 bytes)
    aarch64, 3 registers chain/2: sink moves 0 (rev0), reverse 24
      source: 30 instructions; 12 register moves, 9 stores, 9 loads, 0 slot moves, 2 rematerialized; 8 slots (52 bytes)
      sink: 30 instructions; 12 register moves, 9 stores, 9 loads, 0 slot moves, 2 rematerialized; 8 slots (52 bytes)
    aarch64, 3 registers bmm/0: sink moves 0 (rev0), reverse 32
      source: 38 instructions; 11 register moves, 23 stores, 19 loads, 0 slot moves, 4 rematerialized; 11 slots (76 bytes)
      sink: 38 instructions; 11 register moves, 23 stores, 19 loads, 0 slot moves, 4 rematerialized; 11 slots (76 bytes)
    aarch64, 3 registers softmax/0: sink moves 0 (rev0), reverse 46
      source: 64 instructions; 17 register moves, 41 stores, 37 loads, 0 slot moves, 8 rematerialized; 18 slots (128 bytes)
      sink: 64 instructions; 17 register moves, 41 stores, 37 loads, 0 slot moves, 8 rematerialized; 18 slots (128 bytes)
    aarch64, 3 registers sdpa/0: sink moves 0 (rev0), reverse 101
      source: 127 instructions; 27 register moves, 115 stores, 94 loads, 0 slot moves, 18 rematerialized; 28 slots (204 bytes)
      sink: 127 instructions; 27 register moves, 115 stores, 94 loads, 0 slot moves, 18 rematerialized; 28 slots (204 bytes)
    aarch64, 3 registers conv/0: sink moves 8 (rev1), reverse 84
      source: 105 instructions; 11 register moves, 69 stores, 57 loads, 0 slot moves, 9 rematerialized; 29 slots (188 bytes)
      sink: 105 instructions; 11 register moves, 70 stores, 58 loads, 0 slot moves, 9 rematerialized; 29 slots (188 bytes)
    x86_64, 3 registers chain/0: sink moves 8 (rev1), reverse 63
      source: 92 instructions; 17 register moves, 61 stores, 57 loads, 0 slot moves, 11 rematerialized; 24 slots (156 bytes)
      sink: 92 instructions; 17 register moves, 61 stores, 57 loads, 0 slot moves, 11 rematerialized; 24 slots (156 bytes)
    x86_64, 3 registers chain/1: sink moves 0 (rev0), reverse 48
      source: 55 instructions; 5 register moves, 35 stores, 30 loads, 0 slot moves, 8 rematerialized; 13 slots (92 bytes)
      sink: 55 instructions; 5 register moves, 35 stores, 30 loads, 0 slot moves, 8 rematerialized; 13 slots (92 bytes)
    x86_64, 3 registers chain/2: sink moves 0 (rev0), reverse 30
      source: 36 instructions; 15 register moves, 12 stores, 14 loads, 0 slot moves, 2 rematerialized; 9 slots (60 bytes)
      sink: 36 instructions; 15 register moves, 12 stores, 14 loads, 0 slot moves, 2 rematerialized; 9 slots (60 bytes)
    x86_64, 3 registers bmm/0: sink moves 0 (rev0), reverse 32
      source: 38 instructions; 12 register moves, 22 stores, 22 loads, 0 slot moves, 4 rematerialized; 12 slots (80 bytes)
      sink: 38 instructions; 12 register moves, 22 stores, 22 loads, 0 slot moves, 4 rematerialized; 12 slots (80 bytes)
    x86_64, 3 registers softmax/0: sink moves 8 (rev1), reverse 69
      source: 85 instructions; 28 register moves, 41 stores, 43 loads, 0 slot moves, 14 rematerialized; 17 slots (120 bytes)
      sink: 85 instructions; 28 register moves, 43 stores, 46 loads, 0 slot moves, 14 rematerialized; 17 slots (120 bytes)
    x86_64, 3 registers sdpa/0: sink moves 8 (rev1), reverse 129
      source: 157 instructions; 40 register moves, 105 stores, 94 loads, 0 slot moves, 40 rematerialized; 27 slots (192 bytes)
      sink: 157 instructions; 35 register moves, 108 stores, 97 loads, 0 slot moves, 40 rematerialized; 27 slots (192 bytes)
    x86_64, 3 registers conv/0: sink moves 8 (rev1), reverse 69
      source: 100 instructions; 15 register moves, 71 stores, 68 loads, 0 slot moves, 11 rematerialized; 27 slots (176 bytes)
      sink: 100 instructions; 15 register moves, 71 stores, 68 loads, 0 slot moves, 11 rematerialized; 27 slots (176 bytes) |}]
