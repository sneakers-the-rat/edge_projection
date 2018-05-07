# fig, ax = plt.subplots(tight_layout=True)
# im = ax.imshow(edges_init, cmap=custom_cmap, vmin=0., vmax=1.)
# fig.frameon = False
# ax.axis('off')
# mng = plt.get_current_fig_manager()
# mng.window.state('zoomed')
#plt.show()


# def updatefig(i, display_frame, vid, traces):
#     ret, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # rotate frame 180 degrees
#     #frame = np.rot90(frame, 2)
    
#     # reduce to edges and append to trace list
#     traces.append(imops.scharr_canny(img_as_float(frame), sigma=canny_sig, 
#                                        high_threshold = canny_high,
#                                        low_threshold  = canny_low))
    
#     # iterate through traces setting colors by position
#     #dframe2 = display_frame.copy()
#     display_frame.fill(0.)
#     for trace, color in zip(traces, color_vals):
#         display_frame[trace] = color

#     im.set_array(display_frame)
#     return [im]

# ani = animation.FuncAnimation(fig, updatefig, frames=500, 
#                               interval=5, blit=True, fargs=(display_frame, vid, traces))
# plt.show()
