(* `clone.default`'s [memory_format]: Native's engine has exactly one
   physical layout per shape, so `contiguous_format`/`preserve_format` (and
   omitting the argument) are always already true and just lower; a real
   layout change (`channels_last`/`channels_last_3d`) is refused rather than
   silently treated as a no-op. Every `clone.default` occurrence in the
   100-model sweep requests either no format or exactly `contiguous_format`
   -- see `.ai/pt2_model_support.md`. *)

open Programs

let clone_node mf =
  let arg =
    match mf with
    | None -> ""
    | Some v ->
        jstr
          {|,{"name":"memory_format","arg":{"as_memory_format":%d},"kind":1}|} v
  in
  jstr
    {|{"target":"torch.ops.aten.clone.default","inputs":[{"name":"self","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") arg (as_tensor "y")

let prog mf =
  program ~x_sizes:[ 1; 3; 4; 4 ]
    ~nodes:[ clone_node mf ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

(* MemoryFormat enum values, per schema.yaml: Unknown=0, ContiguousFormat=1,
   ChannelsLast=2, ChannelsLast3d=3, PreserveFormat=4. *)
let%expect_test
    "clone.default: memory_format is a no-op when it can be, rejected otherwise"
    =
  show "absent:" (prog None);
  show "contiguous_format:" (prog (Some 1));
  show "preserve_format:" (prog (Some 4));
  show "channels_last:" (prog (Some 2));
  show "channels_last_3d:" (prog (Some 3));
  show "unknown:" (prog (Some 0));
  [%expect
    {|
    absent:                    lowered, nodes=1
    contiguous_format:         lowered, nodes=1
    preserve_format:           lowered, nodes=1
    channels_last:             malformed PT2 graph: torch.ops.aten.clone.default: memory_format=channels_last is not supported
    channels_last_3d:          malformed PT2 graph: torch.ops.aten.clone.default: memory_format=channels_last_3d is not supported
    unknown:                   malformed PT2 graph: torch.ops.aten.clone.default: memory_format=unknown is not supported |}]

(* A wrong-kind argument under this name is still flagged as a decode
   error, not silently accepted or misreported as an unsupported format. *)
let%expect_test "clone.default: memory_format rejects the wrong argument kind" =
  let node =
    jstr
      {|{"target":"torch.ops.aten.clone.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"memory_format","arg":{"as_int":0},"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor "x") (as_tensor "y")
  in
  show "wrong kind:"
    (program ~x_sizes:[ 1; 3; 4; 4 ] ~nodes:[ node ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {| wrong kind:                malformed PT2 graph: torch.ops.aten.clone.default.memory_format is not an optional memory format |}]
