let pp_space =
  Format.pp_print_list ~pp_sep:(fun f () -> Format.pp_print_char f ' ')

let%expect_test "axis canonical order + pp" =
  Format.printf "%a@." (pp_space Axis.pp) Axis.all;
  [%expect {| N T D H W C |}]

let%expect_test "axis to_int positions" =
  List.iter (fun a -> Format.printf "%a=%d " Axis.pp a (Axis.to_int a)) Axis.all;
  Format.printf "@.";
  [%expect {| N=0 T=1 D=2 H=3 W=4 C=5 |}]
