module A = Machine_rivet_riscv32.Rv32_asm

module Window = struct
  type t = {
    channels : Core.Dim.extent Core.Dim.t;
    in_h : Core.Dim.extent Core.Dim.t;
    in_w : Core.Dim.extent Core.Dim.t;
    kernel_h : Core.Dim.extent Core.Dim.t;
    kernel_w : Core.Dim.extent Core.Dim.t;
    out_channels : Core.Dim.extent Core.Dim.t;
    stride_h : Core.Dim.extent Core.Dim.t;
    stride_w : Core.Dim.extent Core.Dim.t;
  }
end

let i = A.insn
let r = A.reg
let small n = n >= -2048 && n < 2048
let int (d : Core.Dim.extent Core.Dim.t) = (d :> int)

let output_extents (w : Window.t) =
  let axis ~input ~kernel ~stride =
    if int kernel > int stride then Error Kernel_refusal.Overlapping_window
    else if int kernel > int input then Error Kernel_refusal.Window_past_input
    else Ok (Core.Dim.extent (((int input - int kernel) / int stride) + 1))
  in
  Result.bind (axis ~input:w.in_h ~kernel:w.kernel_h ~stride:w.stride_h)
    (fun oh ->
      Result.map
        (fun ow -> (oh, ow))
        (axis ~input:w.in_w ~kernel:w.kernel_w ~stride:w.stride_w))

let symbol (w : Window.t) =
  Printf.sprintf "rvv_window_%dx%dx%d_k%dx%d_s%dx%d_o%d" (int w.in_h)
    (int w.in_w) (int w.channels) (int w.kernel_h) (int w.kernel_w)
    (int w.stride_h) (int w.stride_w) (int w.out_channels)

(* [rd = rs + n] for any 32-bit [n]: an [addi], or through [t6]. *)
let add_from rd rs n =
  if small n then [ i "addi" [ r rd; r rs; A.imm n ] ]
  else A.load_imm "t6" n @ [ i "add" [ r rd; r rs; r "t6" ] ]

let add_const rd n = if n = 0 then [] else add_from rd rd n

let items esc (w : Window.t) =
  let oh, ow =
    match output_extents w with
    | Ok x -> x
    | Error why -> Kernel_refusal.refuse esc why
  in
  let cout = int w.out_channels and cin = int w.channels in
  (* the e32 group that holds the output channels, and its register stride *)
  let lmul, step =
    if cout <= 4 then ("m1", 1)
    else if cout <= 8 then ("m2", 2)
    else if cout <= 16 then ("m4", 4)
    else Kernel_refusal.refuse esc Out_channels_past_16
  in
  let bytes what n =
    if n > 0x7FFF_FFFF then Kernel_refusal.refuse esc (Past_31_bits what) else n
  in
  let pixel = bytes "a pixel" (4 * cin) in
  let row = bytes "an input row" (pixel * int w.in_w) in
  ignore (bytes "the input" (row * int w.in_h));
  ignore (bytes "the output" (4 * cout * int oh * int ow));
  ignore
    (bytes "the weights" (4 * cout * cin * int w.kernel_h * int w.kernel_w));
  let v k = Printf.sprintf "v%d" (k * step) in
  let outer = v 1 and row_acc = v 2 and col = v 3 and tap = v 4 in
  let name = symbol w in
  let lab s = name ^ "_" ^ s in
  let taps =
    List.concat
      (List.init (int w.kernel_h) (fun kh ->
           [ i "vmv.v.i" [ r col; A.imm 0 ] ]
           @ List.concat
               (List.init (int w.kernel_w) (fun kw ->
                    let off = bytes "a tap" ((kh * row) + (kw * pixel)) in
                    let base, at =
                      if small off then ("t2", off) else ("t1", 0)
                    in
                    (if small off then [] else add_from "t1" "t2" off)
                    @ [
                        i "flw" [ r "ft0"; A.mem base at ];
                        i "vle32.v" [ r tap; A.mem "t3" 0 ];
                      ]
                    @ add_const "t3" (4 * cout)
                    @ [
                        i "vfmul.vf" [ r tap; r tap; r "ft0" ];
                        i "vfadd.vv" [ r col; r col; r tap ];
                      ]))
           @ [ i "vfadd.vv" [ r row_acc; r row_acc; r col ] ]))
  in
  [ A.text; A.align 4; A.global name; A.label name ]
  @ A.load_imm "t0" cout
  @ [
      i "vsetvli"
        [ r "zero"; r "t0"; A.sym "e32"; A.sym lmul; A.sym "ta"; A.sym "ma" ];
    ]
  @ A.load_imm "a4" (int oh)
  @ [ A.label (lab "oh"); i "mv" [ r "a7"; r "a0" ] ]
  @ A.load_imm "a5" (int ow)
  @ [
      A.label (lab "ow");
      i "mv" [ r "t2"; r "a7" ];
      i "mv" [ r "t3"; r "a1" ];
      i "vmv.v.i" [ r outer; A.imm 0 ];
    ]
  @ A.load_imm "a6" cin
  @ [ A.label (lab "c"); i "vmv.v.i" [ r row_acc; A.imm 0 ] ]
  @ taps
  @ [
      i "vfadd.vv" [ r outer; r outer; r row_acc ];
      i "addi" [ r "t2"; r "t2"; A.imm 4 ];
      i "addi" [ r "a6"; r "a6"; A.imm (-1) ];
      i "bnez" [ r "a6"; A.sym (lab "c") ];
      i "vle32.v" [ r tap; A.mem "a2" 0 ];
      i "vfadd.vv" [ r outer; r outer; r tap ];
      i "vse32.v" [ r outer; A.mem "a3" 0 ];
    ]
  @ add_const "a3" (4 * cout)
  @ add_const "a7" (int w.stride_w * pixel)
  @ [
      i "addi" [ r "a5"; r "a5"; A.imm (-1) ];
      i "bnez" [ r "a5"; A.sym (lab "ow") ];
    ]
  @ add_const "a0" (int w.stride_h * row)
  @ [
      i "addi" [ r "a4"; r "a4"; A.imm (-1) ];
      i "bnez" [ r "a4"; A.sym (lab "oh") ];
      i "li" [ r "a0"; A.imm 0 ];
      i "ret" [];
    ]

let module_ w =
  Kernel_refusal.generate @@ fun esc ->
  let items = items esc w in
  match A.module_ (symbol w) (fun () -> items) with
  | Ok m -> m
  | Error what -> Kernel_refusal.refuse esc (Assembler what)
