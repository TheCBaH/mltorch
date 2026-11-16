(* Sizes are derived from the shapes the design names, in binary32 (4 bytes an
   element), so a figure that drifts from its shape fails to type-check as
   data. *)

let f32 = 4L
let ( * ) = Int64.mul
let bytes n = Data_bytes.of_int64 n
let align = bytes 16L

let alloc name placement n : Image.Allocation.t =
  { align; bytes = bytes n; name; placement }

(* 8 -> 48 pointwise with its Hardtanh at 112x112, two output rows a block. *)
let pointwise_mobilenetv2_050 ~code : Image.t =
  let side = 112L in
  {
    name = "mobilenetv2_050 pointwise 8 -> 48 at 112x112";
    code;
    allocations =
      [
        alloc "input" In (side * side * 8L * f32);
        alloc "output" Out (side * side * 48L * f32);
        alloc "scratch" Scratch (2L * side * 48L * f32);
      ];
  }

(* Expand 16 -> 96, Silu, depthwise 3x3 stride 2, Silu: the 112x112 expansion
   is a band of four rows, two-row blocks of scratch; the output is 56x56. *)
let expand_depthwise_efficientnet_b0 ~code : Image.t =
  let side = 112L in
  {
    name = "efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2";
    code;
    allocations =
      [
        alloc "input" In (side * side * 16L * f32);
        alloc "output" Out (56L * 56L * 96L * f32);
        alloc "band" Scratch (4L * side * 96L * f32);
        alloc "scratch" Scratch (2L * side * 96L * f32);
      ];
  }

let all ~code =
  [ pointwise_mobilenetv2_050 ~code; expand_depthwise_efficientnet_b0 ~code ]
