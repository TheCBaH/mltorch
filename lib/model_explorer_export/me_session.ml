(* Visualization_session v1. See the .mli.

   This file is now a thin facade: the pane/comparison/flow wire types live
   in me_session_panes.ml, Capability/View in me_session_capability.ml, and
   the Session document itself in me_session_document.ml.. *)

module Producer = Me_session_panes.Producer
module Model_summary = Me_session_panes.Model_summary
module Pane_state = Me_session_panes.Pane_state
module Mapping_entry = Me_session_panes.Mapping_entry
module Sync_navigation = Me_session_panes.Sync_navigation
module Flow_state_view = Me_session_panes.Flow_state_view
module Flow_view_graph = Me_session_panes.Flow_view_graph
module Flow_destination = Me_session_panes.Flow_destination
module Comparison = Me_session_panes.Comparison
module Node_data_set = Me_session_panes.Node_data_set
module Capability = Me_session_capability.Capability
module View = Me_session_capability.View
module Session = Me_session_document.Session

module Runtime = struct
  type t = { epoch : string; limits : Me_limits.Limits.t; session : Session.t }
end
