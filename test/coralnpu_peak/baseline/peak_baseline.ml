(* One line per peak kernel: every segment's size and MD5, the entry and every
   export as an offset from the fixed bind base. The cases are peak.t's, which
   lists the same 47; keep the two in step. [listing] prints a kernel's
   instructions for investigating a difference (dune top
   test/coralnpu_peak/baseline). *)

module K = Peak_kernels
module Rv32_image = Machine_rivet_riscv32.Rv32_image

let base = 0x10000L

(* [name vtype instruction], [name tile instruction], or a matrix multiply
   ([gemm_...] through RVV, [tgemm_...] through Zvt) with no operands. *)
type case =
  | Elementwise of string * int * int * string
  | Gemm of K.Gemm.Input.t * int option
      (** and its width, if not the default *)
  | Tile of string * string
  | Tile_gemm of Peak_zvt.Tile_gemm.Input.t * Peak_zvt.Tile_gemm.Geometry.t

let elementwise name sew lmul instr = Elementwise (name, sew, lmul, instr)

let cases =
  [
    elementwise "i8add" 8 4 "vadd.vv";
    elementwise "i8mul" 8 4 "vmul.vv";
    elementwise "i8macc" 8 4 "vmacc.vv";
    elementwise "i8wmacc" 8 2 "vwmacc.vv";
    elementwise "i16add" 16 4 "vadd.vv";
    elementwise "i16mul" 16 4 "vmul.vv";
    elementwise "i16macc" 16 4 "vmacc.vv";
    elementwise "i16wmacc" 16 2 "vwmacc.vv";
    elementwise "i32add" 32 4 "vadd.vv";
    elementwise "i32mul" 32 4 "vmul.vv";
    elementwise "i32macc" 32 4 "vmacc.vv";
    elementwise "f32add" 32 4 "vfadd.vv";
    elementwise "f32mul" 32 4 "vfmul.vv";
    elementwise "f32macc" 32 4 "vfmacc.vv";
    elementwise "bf16wmacc" 16 2 "vfwmaccbf16.vv";
    elementwise "f32macc_m1" 32 1 "vfmacc.vv";
    Gemm (I8, None);
    Gemm (I16, None);
    Gemm (I32, None);
    Gemm (F32, None);
    Gemm (Bf16, None);
    Gemm (I8, Some 48);
    Gemm (I8, Some 128);
    Gemm (I16, Some 24);
    Gemm (I16, Some 64);
    Gemm (I32, Some 24);
    Gemm (I32, Some 64);
    Gemm (F32, Some 24);
    Gemm (F32, Some 64);
    Gemm (Bf16, Some 24);
    Gemm (Bf16, Some 64);
    Tile ("tile_f32", "vtfmm.tvv");
    Tile ("tile_bf16", "vtfmm.alt.tvv");
    Tile ("tile_i8u", "vtmmu.tvv");
    Tile ("tile_i8s", "vtmms.tvv");
    Tile_gemm (I8, Engine);
    Tile_gemm (Bf16, Engine);
    Tile_gemm (F32, Engine);
    Tile_gemm (I8, Rvv { width = None });
    Tile_gemm (I8, Rvv { width = Some 48 });
    Tile_gemm (I8, Rvv { width = Some 128 });
    Tile_gemm (Bf16, Rvv { width = None });
    Tile_gemm (Bf16, Rvv { width = Some 24 });
    Tile_gemm (Bf16, Rvv { width = Some 64 });
    Tile_gemm (F32, Rvv { width = None });
    Tile_gemm (F32, Rvv { width = Some 24 });
    Tile_gemm (F32, Rvv { width = Some 64 });
  ]

let name = function
  | Elementwise (n, _, _, _) | Tile (n, _) -> n
  | Gemm (g, w) ->
      "gemm_" ^ K.Gemm.Input.name g
      ^ Option.fold ~none:"" ~some:(Printf.sprintf "_w%d") w
  | Tile_gemm (t, g) -> (
      "tgemm_"
      ^ Peak_zvt.Tile_gemm.Input.name t
      ^
      match g with
      | Engine -> ""
      | Rvv { width } ->
          "_rvv" ^ Option.fold ~none:"" ~some:(Printf.sprintf "_w%d") width)

let kernel = function
  | Elementwise (_, sew, lmul, instr) -> (
      match K.Vv_op.of_mnemonic instr with
      | None -> failwith ("no vector-vector form " ^ instr)
      | Some op -> (
          match K.elementwise ~sew ~lmul op with
          | Ok k -> k
          | Error e -> failwith e))
  | Gemm (g, width) -> K.gemm ?width g
  | Tile (_, instr) -> (
      match K.Insn.Tile_op.of_mnemonic instr with
      | None -> failwith ("no tile multiply " ^ instr)
      | Some op -> Peak_zvt.tile op)
  | Tile_gemm (t, geometry) -> Peak_zvt.tile_gemm ~geometry t

let module_ k =
  match K.build k K.rv32_profile with Ok m -> m | Error e -> failwith e

let bound ?entry m =
  let entry = Option.value entry ~default:K.entry in
  let laid =
    Err.payload (Rv32_image.plan ~profile:K.rv32_profile ~entry [ m ])
  in
  match laid with
  | Error e -> Fmt.failwith "plan: %a" Rv32_image.Error.pp e
  | Ok laid -> (
      match Err.payload (Rv32_image.bind ~base laid) with
      | Ok image -> image
      | Error e -> Fmt.failwith "bind: %a" Rv32_image.Error.pp e)

let offset a = Int64.to_int (Int64.sub a base)

let line case =
  let image = bound (module_ (kernel case)) in
  let segments =
    List.map
      (fun (s : Image.segment) ->
        Fmt.str "%s:%d:%s" s.Image.name
          (String.length s.Image.bytes)
          (Digest.to_hex (Digest.string s.Image.bytes)))
      image.Image.segments
  in
  let exports =
    List.sort compare image.Image.exports
    |> List.map (fun (n, a) -> Fmt.str "%s=+0x%x" n (offset a))
  in
  let entry =
    match image.Image.entry with
    | None -> "none"
    | Some a ->
        if List.assoc_opt K.entry image.Image.exports <> Some a then
          failwith (name case ^ ": entry is not the exported peak_entry");
        Fmt.str "+0x%x" (offset a)
  in
  Fmt.str "%s seg=[%s] entry=%s exports=[%s]" (name case)
    (String.concat "; " segments)
    entry
    (String.concat "; " exports)

let print () = List.iter (fun c -> print_endline (line c)) cases

(* Rivet's rendering of each item, for the toplevel. *)
let listing case =
  List.iter
    (function
      | Asm_core.Normalized_ast.Instruction { insn; _ } ->
          Fmt.pr "  %a@." Riscv32_encode.Instruction.pp insn
      | Label { name; _ } -> Fmt.pr "%s:@." name
      | Directive _ -> ())
    (module_ (kernel case)).Asm_core.Normalized_ast.items

let%expect_test "peak kernels encode to the pinned bytes" =
  print ();
  [%expect
    {|
    i8add seg=[.text:112:a9a166e7df756a5e0cd9bd8971259936] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i8mul seg=[.text:112:37dd39eb6b85085bf72df8c18810df5e] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i8macc seg=[.text:112:bf5b2cbcac1c2875e8817de574e91381] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i8wmacc seg=[.text:112:6ce777243faccadfcea6190232e37d59] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i16add seg=[.text:112:2358d11385bfea00b6adce7446794885] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i16mul seg=[.text:112:cb2edce45f58556f9751dd35bc857854] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i16macc seg=[.text:112:3744fbbf4f307292d21bbf2ea3c80c95] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i16wmacc seg=[.text:112:982ee12e3678c7983a97037b73c47aa8] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i32add seg=[.text:112:03111ae58491dabf852e887e7d30b35c] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i32mul seg=[.text:112:7e7f49c5d8f9d4bf1cbec1140a5fefe1] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    i32macc seg=[.text:112:b53f2f8c49f9689d59bf357329434d2c] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    f32add seg=[.text:112:d536667db1cee75df68b88e4badea686] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    f32mul seg=[.text:112:c7de70e1a7422fe4cb1a0b1f69f202e7] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    f32macc seg=[.text:112:c615151381f91285a2c8bc9e1b6211e7] entry=+0x0 exports=[peak_body=+0x2c; peak_body_end=+0x44; peak_entry=+0x0]
    bf16wmacc seg=[.text:116:05934c1b0817962f15e92df2b737bcb1] entry=+0x0 exports=[peak_body=+0x30; peak_body_end=+0x48; peak_entry=+0x0]
    f32macc_m1 seg=[.text:304:f4fe11642c5c2dea1ea23e395c547049] entry=+0x0 exports=[peak_body=+0x5c; peak_body_end=+0xa4; peak_entry=+0x0]
    gemm_i8 seg=[.text:244:3d84d83347e3187d02aa3865bc7de720] entry=+0x0 exports=[peak_body=+0x48; peak_body_end=+0xb4; peak_entry=+0x0]
    gemm_i16 seg=[.text:296:51e87c903251c9a807ef927db380a5a9] entry=+0x0 exports=[peak_body=+0x54; peak_body_end=+0xe0; peak_entry=+0x0]
    gemm_i32 seg=[.text:196:4660b1380f7595fc9aec2cbd893481d5] entry=+0x0 exports=[peak_body=+0x44; peak_body_end=+0x90; peak_entry=+0x0]
    gemm_f32 seg=[.text:196:62093da79fbeb6a368fdaae1536482ee] entry=+0x0 exports=[peak_body=+0x44; peak_body_end=+0x90; peak_entry=+0x0]
    gemm_bf16 seg=[.text:232:5a5b3db0fcc1bd5730a5ad8002cab211] entry=+0x0 exports=[peak_body=+0x54; peak_body_end=+0xa0; peak_entry=+0x0]
    gemm_i8_w48 seg=[.text:244:675001b25189ae01fa5ef9c4eaa86632] entry=+0x0 exports=[peak_body=+0x48; peak_body_end=+0xb4; peak_entry=+0x0]
    gemm_i8_w128 seg=[.text:288:e64b94d4822840475feef5d3f5ca08a1] entry=+0x0 exports=[peak_body=+0x58; peak_body_end=+0xc4; peak_entry=+0x0]
    gemm_i16_w24 seg=[.text:296:0ea478b204a3208bdec1a9ed3e09e454] entry=+0x0 exports=[peak_body=+0x54; peak_body_end=+0xe0; peak_entry=+0x0]
    gemm_i16_w64 seg=[.text:356:270768ae2e3ab69dced4ceeac5cad98c] entry=+0x0 exports=[peak_body=+0x64; peak_body_end=+0xf0; peak_entry=+0x0]
    gemm_i32_w24 seg=[.text:196:5061a52d75c2ca2a8e8eb12ce123528c] entry=+0x0 exports=[peak_body=+0x44; peak_body_end=+0x90; peak_entry=+0x0]
    gemm_i32_w64 seg=[.text:256:e394869ec2412063c08ffe018b77f7bb] entry=+0x0 exports=[peak_body=+0x54; peak_body_end=+0xa0; peak_entry=+0x0]
    gemm_f32_w24 seg=[.text:196:acc17754aa524d2c01b8925e86b75036] entry=+0x0 exports=[peak_body=+0x44; peak_body_end=+0x90; peak_entry=+0x0]
    gemm_f32_w64 seg=[.text:256:7bfdc92c39c30a7ac30234199d3baa6f] entry=+0x0 exports=[peak_body=+0x54; peak_body_end=+0xa0; peak_entry=+0x0]
    gemm_bf16_w24 seg=[.text:232:0a9561c4b270c5e8d3b4849a8828fc52] entry=+0x0 exports=[peak_body=+0x54; peak_body_end=+0xa0; peak_entry=+0x0]
    gemm_bf16_w64 seg=[.text:292:285ea3a671907e8d84adaf342d0c6b5d] entry=+0x0 exports=[peak_body=+0x64; peak_body_end=+0xb0; peak_entry=+0x0]
    tile_f32 seg=[.text:232:cac82aa07b9354433daea8da1b48d060] entry=+0x0 exports=[peak_body=+0x40; peak_body_end=+0x58; peak_entry=+0x0]
    tile_bf16 seg=[.text:248:264a6ec96527d668427a8c89e39fe685] entry=+0x0 exports=[peak_body=+0x50; peak_body_end=+0x68; peak_entry=+0x0]
    tile_i8u seg=[.text:256:2291f3145973912ef38aadada49fd4f1] entry=+0x0 exports=[peak_body=+0x58; peak_body_end=+0x70; peak_entry=+0x0]
    tile_i8s seg=[.text:256:aa949c45e67ea9c4bc80e5a24ae952c0] entry=+0x0 exports=[peak_body=+0x58; peak_body_end=+0x70; peak_entry=+0x0]
    tgemm_i8 seg=[.text:416:b6e9d30b7adf7d50f01775844f1b308a] entry=+0x0 exports=[peak_body=+0x68; peak_body_end=+0xd0; peak_entry=+0x0]
    tgemm_bf16 seg=[.text:416:34479ad8fdcdc6ca4162dc748f065d13] entry=+0x0 exports=[peak_body=+0x68; peak_body_end=+0xd0; peak_entry=+0x0]
    tgemm_f32 seg=[.text:416:e0e87e76a5d04dc17d00c5908c785f65] entry=+0x0 exports=[peak_body=+0x68; peak_body_end=+0xd0; peak_entry=+0x0]
    tgemm_i8_rvv seg=[.text:676:021651b235c1c655fdfff32a5dce2879] entry=+0x0 exports=[peak_body=+0x70; peak_body_end=+0xe8; peak_entry=+0x0]
    tgemm_i8_rvv_w48 seg=[.text:680:86e7387b5bf69b7ae927c5992a72d581] entry=+0x0 exports=[peak_body=+0x70; peak_body_end=+0xe8; peak_entry=+0x0]
    tgemm_i8_rvv_w128 seg=[.text:1300:56e47cdafbf553b1da41830cfb8596e7] entry=+0x0 exports=[peak_body=+0x70; peak_body_end=+0xe8; peak_entry=+0x0]
    tgemm_bf16_rvv seg=[.text:312:20eaae44738b18daa884a1b7affc1518] entry=+0x0 exports=[peak_body=+0x60; peak_body_end=+0xb8; peak_entry=+0x0]
    tgemm_bf16_rvv_w24 seg=[.text:344:48361291331dc61a5c2f415eca910c97] entry=+0x0 exports=[peak_body=+0x64; peak_body_end=+0xcc; peak_entry=+0x0]
    tgemm_bf16_rvv_w64 seg=[.text:576:6f3dcc4bfdfab2ebcc10c3856690d1ee] entry=+0x0 exports=[peak_body=+0x60; peak_body_end=+0xb8; peak_entry=+0x0]
    tgemm_f32_rvv seg=[.text:288:8a422e6f8b78cb509895d8cd1f9ef302] entry=+0x0 exports=[peak_body=+0x58; peak_body_end=+0xa0; peak_entry=+0x0]
    tgemm_f32_rvv_w24 seg=[.text:320:f9d446cef23757f071b675ef22f0b64d] entry=+0x0 exports=[peak_body=+0x5c; peak_body_end=+0xb4; peak_entry=+0x0]
    tgemm_f32_rvv_w64 seg=[.text:528:f02ac613bcee0c7cd080adafbf8e3a1f] entry=+0x0 exports=[peak_body=+0x58; peak_body_end=+0xa0; peak_entry=+0x0] |}]
