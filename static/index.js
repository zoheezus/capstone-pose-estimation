// wait for the DOM to be loaded
$(document).ready(function () {
  // bind 'myForm' and provide a simple callback function
  $("button#start").click(function (e) {
    e.preventDefault();
    var button = $(this);
    $.ajax({
      type: "POST",
      url: "/start",
      // data: {
      //   server_add: $("select#server_add").find(":selected").text(), // < note use of 'this' here
      // },
      data: {},
      success: function (result) {
        // this part will not load until all data is loaded. therefore it does
        // not work for streaming
        //                $("#process_container").text(result)
        toastr["success"](result);
        $("#process_container").prepend(
          '<img id="camera_feed" class="mx-auto d-block" src="/camera_feed' +
            "?" +
            Math.random() +
            '"/>'
        );
        button.prop("disabled", true);
      },
      error: function (result) {
        toastr["error"]("An error occurred.");
        $("#process_container").text(result);
        if ($("#camera_feed").length > 0) {
          $("#camera_feed").remove();
        }
      },
    });
  });

  $("button#stop").click(function (e) {
    e.preventDefault();
    $.ajax({
      type: "POST",
      url: "/stop",
      data: {},
      success: function (result) {
        if ($("#camera_feed").length > 0) {
          $("#camera_feed").remove();
        }
        toastr["success"](result);
        $("button#start").prop("disabled", false);
      },
      error: function (result) {
        toastr["error"](result);
      },
    });
  });
});
