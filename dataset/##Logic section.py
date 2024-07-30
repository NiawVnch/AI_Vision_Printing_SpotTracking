        ##Logic section
        # global spots_status1 
        global spots_status2
        # global previous_spots_status1
        global previous_spots_status2

        not_empty_count2 = sum(1 for s in spots_status2 if s == NOT_EMPTY)
        # not_empty_count1 = sum(1 for s in spots_status1 if s == NOT_EMPTY)

        if not e_detected and not_empty_count2 > sum(1 for s in previous_spots_status2 if s == NOT_EMPTY):
            t_E = time.time() - start_time
            e_detected = True
            LstStat_LogicLoop = "Good"
            print("E stage detected at: {:.1f} seconds".format(t_E))
        
            if t_prev_E is not None:
                cycle_time = (t_E - t_prev_E)
                e_detected = False
                print("Cycle time: {}".format(cycle_time))
                ##MQTT section
                # Create a client instance and specify the MQTT protocol version
                client.on_connect = on_connect
                client.on_publish = on_publish

                try:
                    # Replace "localhost" with the broker's IP if your broker is not on the local machine
                    client.connect(Host_Addr, firewall_port, 60)

                    # Start a loop to process callbacks and manage reconnections, then stop after publishing
                    client.loop_start()
                    time.sleep(1)  # Give some time for the connection and publication
                    client.loop_stop()
                    client.disconnect()
                except Exception as e:
                    print(f"An error occurred: {e}")

            else:
                e_detected = False

        t_prev_E = t_E