open Machine_rivet_riscv32
module R = Rv32_kernel_reg

let all_registers =
  List.map (fun r -> (Fmt.str "%a" R.Gpr.pp r, R.Gpr.to_rivet r)) R.Gpr.all
  @ List.map (fun r -> (Fmt.str "%a" R.Fpr.pp r, R.Fpr.to_rivet r)) R.Fpr.all
  @ List.map (fun r -> (Fmt.str "%a" R.Vreg.pp r, R.Vreg.to_rivet r)) R.Vreg.all

(* A mistyped ABI name, or a number out of place, makes Rivet's own lookup of the
   printed name disagree. *)
let%expect_test "every register's name resolves to its own Rivet register" =
  Fmt.pr "%d registers@." (List.length all_registers);
  List.iter
    (fun (name, rivet) ->
      if Riscv32_encode.Reg.find name <> Some rivet then
        Fmt.pr "%s disagrees with Rivet@." name)
    all_registers;
  [%expect {| 96 registers |}]

let%expect_test "of_index is checked, and numbered in order" =
  let show pp r =
    match Err.payload r with
    | Ok r -> Fmt.pr "%a@." pp r
    | Error (`Register_index i) -> Fmt.pr "%a@." R.Register_index.pp i
  in
  List.iter (fun i -> show R.Gpr.pp (R.Gpr.of_index i)) [ -1; 0; 31; 32 ];
  show R.Fpr.pp (R.Fpr.of_index 10);
  show R.Vreg.pp (R.Vreg.of_index 32);
  Fmt.pr "%b@."
    (List.for_all2
       (fun i r -> R.Gpr.index r = i)
       (List.init 32 Fun.id) R.Gpr.all);
  [%expect
    {|
    there is no integer register -1
    zero
    t6
    there is no integer register 32
    fa0
    there is no vector register 32
    true |}]
