(* The identity a Native runner reports is the one for its dot accumulation, so
   a binary32-sequential run is never recorded as a binary64 one. *)

module N = Inference_compare_pt2.Native_runner

let identity_of dots =
  let module R = (val N.create ~dots ()) in
  R.identity ()

let%expect_test "the runner's identity follows its dot accumulation" =
  let b64 = identity_of Direct.Binary64
  and b32 = identity_of Direct.Binary32_sequential in
  Printf.printf "runner: binary64 = factory %b; sequential differs %b\n"
    (b64 = N.identity ~dots:Direct.Binary64 ())
    (b64 <> b32);
  Printf.printf "runner: sequential = factory %b\n"
    (b32 = N.identity ~dots:Direct.Binary32_sequential ());
  [%expect
    {|
    runner: binary64 = factory true; sequential differs true
    runner: sequential = factory true
    |}]
