Machine-readable support-matrix snapshot for the PT2-functional-release models
(data/pt2-functional/, fetched by `make pt2.download-cram` + `make pt2.download
PT2_MODEL=csatv2`). See .ai/pt2_model_support.md for how to read/refresh this and
what each column means; that doc is prose over the same underlying commands, this
is the pinned, parseable golden. Gated on PT2_DATA; run with `make pt2.runtest`.

One compact JSON object per line (JSON Lines), so CI or a script can consume it
with `jq` without depending on this file's prose staying in sync. Fields:
`model` (the archive name); `native_builds` (does `to4d`'s first stage --
canonical native import + fold -- succeed at all? false means some node in
the graph has no Native lowering yet, a missing operation rather than a
domain limit); `native4d_converts` (does the WHOLE canonical graph fit the
Native4D four-axis dialect? null when `native_builds` is false -- the
question doesn't apply yet -- and false with `native_builds` true means a
genuine four-axis domain limit, see .ai/native4d_design.md, not missing
work); `nodes` (canonical native node count, null when `native_builds` is
false); `blocker` (null on full success, otherwise the first line native
graph construction failed on, or `to4d`'s own "outside the dialect"
message).

  $ for m in mobilenetv2_050 regnetx_002 efficientnet_b0 test_convnext2 fastvit_sa12 csatv2; do
  >   out=$(../bin/native_graph.exe to4d --pt2 "$PT2_DATA/$m/$m.pt2" --fold 2>&1)
  >   status=$?
  >   line1=$(printf '%s\n' "$out" | sed -n '1p')
  >   line2=$(printf '%s\n' "$out" | sed -n '2p')
  >   esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
  >   if [ "$status" -ne 0 ]; then
  >     printf '{"model":"%s","native_builds":false,"native4d_converts":null,"nodes":null,"blocker":"%s"}\n' \
  >       "$m" "$(esc "$line1")"
  >   else
  >     nodes=$(printf '%s' "$line1" | sed -n 's/^canonical native: nodes=\([0-9]*\)$/\1/p')
  >     case "$line2" in
  >       native4d:*)
  >         printf '{"model":"%s","native_builds":true,"native4d_converts":true,"nodes":%s,"blocker":null}\n' \
  >           "$m" "$nodes"
  >         ;;
  >       "outside the dialect: "*)
  >         reason=$(printf '%s' "$line2" | sed -n 's/^outside the dialect: //p')
  >         printf '{"model":"%s","native_builds":true,"native4d_converts":false,"nodes":%s,"blocker":"%s"}\n' \
  >           "$m" "$nodes" "$(esc "$reason")"
  >         ;;
  >       *)
  >         printf '{"model":"%s","native_builds":true,"native4d_converts":null,"nodes":%s,"blocker":"%s"}\n' \
  >           "$m" "$nodes" "$(esc "unrecognized to4d output: $line2")"
  >         ;;
  >     esac
  >   fi
  > done
  {"model":"mobilenetv2_050","native_builds":true,"native4d_converts":true,"nodes":100,"blocker":null}
  {"model":"regnetx_002","native_builds":true,"native4d_converts":false,"nodes":101,"blocker":"node n373: convolution has 3 groups, which is neither 1 nor depthwise"}
  {"model":"efficientnet_b0","native_builds":true,"native4d_converts":true,"nodes":254,"blocker":null}
  {"model":"test_convnext2","native_builds":true,"native4d_converts":true,"nodes":49,"blocker":null}
  {"model":"fastvit_sa12","native_builds":true,"native4d_converts":false,"nodes":366,"blocker":"node n604: axis T is outside the N/H/W/C dialect"}
  {"model":"csatv2","native_builds":false,"native4d_converts":null,"nodes":null,"blocker":"native_graph: unsupported PT2 operator: torch.ops.aten.index.Tensor"}
