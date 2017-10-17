def train(F,
          num_epochs=200,  # number of epochs
          learning_rate=0.0002,  # learning rate of optimizer
          beta1=0.5,  # parameter for Adam optimizer
          decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
          enable_shuffle=True,  # enable shuffle of the dataset
          use_trained_model=True,  # used the saved checkpoint to initialize the model
          ):
    # *************************** load file names of images ******************************************************
    file_names = glob(os.path.join('./data', F.dataset_name, '*.jpg'))
    size_data = len(file_names)
    np.random.seed(seed=2017)
    if enable_shuffle:
        np.random.shuffle(file_names)

    # *********************************** optimizer **************************************************************
    # over all, there are three loss functions, weights may differ from the paper because of different datasets
    F.loss_EG = F.EG_loss + 0.000 * F.G_img_loss + 0.000 * F.E_z_loss + 0.000 * F.tv_loss  # slightly increase the params  
    F.loss_Dz = F.D_z_loss_prior + F.D_z_loss_z
    F.loss_Di = F.D_img_loss_input + F.D_img_loss_G

    # set learning rate decay
    F.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
    EG_learning_rate = tf.train.exponential_decay(
        learning_rate=learning_rate,
        global_step=F.EG_global_step,
        decay_steps=size_data / F.size_batch * 2,
        decay_rate=decay_rate,
        staircase=True
    )

    # optimizer for encoder + generator

    F.EG_optimizer = tf.train.AdamOptimizer(
        learning_rate=EG_learning_rate,
        beta1=beta1
    ).minimize(
        loss=F.loss_EG,
        global_step=F.EG_global_step,
        var_list=F.E_variables + F.G_variables
    )

    # optimizer for discriminator on z
    F.D_z_optimizer = tf.train.AdamOptimizer(
        learning_rate=EG_learning_rate,
        beta1=beta1
    ).minimize(
        loss=F.loss_Dz,
        var_list=F.D_z_variables
    )

    # optimizer for discriminator on image
    F.D_img_optimizer = tf.train.AdamOptimizer(
        learning_rate=EG_learning_rate,
        beta1=beta1
    ).minimize(
        loss=F.loss_Di,
        var_list=F.D_img_variables
    )

    # *********************************** tensorboard *************************************************************
    # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
    F.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
    F.summary = tf.summary.merge([
        F.z_summary, F.z_prior_summary,
        F.D_z_loss_z_summary, F.D_z_loss_prior_summary,
        F.D_z_logits_summary, F.D_z_prior_logits_summary,
        F.EG_loss_summary, F.E_z_loss_summary,
        F.D_img_loss_input_summary, F.D_img_loss_G_summary,
        F.G_img_loss_summary, F.EG_learning_rate_summary,
        F.D_G_logits_summary, F.D_input_logits_summary
    ])
    F.writer = tf.summary.FileWriter(os.path.join(F.save_dir, 'summary'), F.session.graph)

    # ************* get some random samples as testing data to visualize the learning process *********************
    sample_files = file_names[0:F.size_batch]
    file_names[0:F.size_batch] = []
    sample = [load_image(
        image_path=sample_file,
        image_size=F.size_image,
        image_value_range=F.image_value_range,
        is_gray=(F.num_input_channels == 1),
    ) for sample_file in sample_files]
    if F.num_input_channels == 1:
        sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
        sample_images = np.array(sample).astype(np.float32)
    sample_label_age = np.ones(
        shape=(len(sample_files), F.num_categories),
        dtype=np.float32
    ) * F.image_value_range[0]
    sample_label_gender = np.ones(
        shape=(len(sample_files), 2),
        dtype=np.float32
    ) * F.image_value_range[0]
    for i, label in enumerate(sample_files):
        label = int(str(sample_files[i]).split('\\')[-1].split('_')[0])
        if 0 <= label <= 5:
            label = 0
        elif 6 <= label <= 10:
            label = 1
        elif 11 <= label <= 15:
            label = 2
        elif 16 <= label <= 20:
            label = 3
        elif 21 <= label <= 30:
            label = 4
        elif 31 <= label <= 40:
            label = 5
        elif 41 <= label <= 50:
            label = 6
        elif 51 <= label <= 60:
            label = 7
        elif 61 <= label <= 70:
            label = 8
        else:
            label = 9
        sample_label_age[i, label] = F.image_value_range[-1]
        gender = int(str(sample_files[i]).split('\\')[-1].split('_')[1])
        sample_label_gender[i, gender] = F.image_value_range[-1]

    # ******************************************* training *******************************************************
    print('\n\tPreparing for training ...')

    # initialize the graph
    tf.global_variables_initializer().run()

    # load check point
    if use_trained_model:
        if F.load_checkpoint():
            print("\tSUCCESS ^_^")
        else:
            print("\tFAILED >_<!")

    # epoch iteration
    num_batches = len(file_names) // F.size_batch
    for epoch in range(num_epochs):
        if enable_shuffle:
            np.random.shuffle(file_names)
        for ind_batch in range(num_batches):
            start_time = time.time()
            # read batch images and labels
            batch_files = file_names[ind_batch * F.size_batch:(ind_batch + 1) * F.size_batch]
            batch = [load_image(
                image_path=batch_file,
                image_size=F.size_image,
                image_value_range=F.image_value_range,
                is_gray=(F.num_input_channels == 1),
            ) for batch_file in batch_files]
            if F.num_input_channels == 1:
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)
            batch_label_age = np.ones(
                shape=(len(batch_files), F.num_categories),
                dtype=np.float
            ) * F.image_value_range[0]
            batch_label_gender = np.ones(
                shape=(len(batch_files), 2),
                dtype=np.float
            ) * F.image_value_range[0]
            for i, label in enumerate(batch_files):
                label = int(str(batch_files[i]).split('/')[-1].split('_')[0])
                if 0 <= label <= 5:
                    label = 0
                elif 6 <= label <= 10:
                    label = 1
                elif 11 <= label <= 15:
                    label = 2
                elif 16 <= label <= 20:
                    label = 3
                elif 21 <= label <= 30:
                    label = 4
                elif 31 <= label <= 40:
                    label = 5
                elif 41 <= label <= 50:
                    label = 6
                elif 51 <= label <= 60:
                    label = 7
                elif 61 <= label <= 70:
                    label = 8
                else:
                    label = 9
                batch_label_age[i, label] = F.image_value_range[-1]
                gender = int(str(batch_files[i]).split('\\')[-1].split('_')[1])
                batch_label_gender[i, gender] = F.image_value_range[-1]

            # prior distribution on the prior of z
            batch_z_prior = np.random.uniform(
                F.image_value_range[0],
                F.image_value_range[-1],
                [F.size_batch, F.num_z_channels]
            ).astype(np.float32)

            # update
            _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, TV = F.session.run(
                fetches=[
                    F.EG_optimizer,
                    F.D_z_optimizer,
                    F.D_img_optimizer,
                    F.EG_loss,
                    F.E_z_loss,
                    F.D_z_loss_z,
                    F.D_z_loss_prior,
                    F.G_img_loss,
                    F.D_img_loss_G,
                    F.D_img_loss_input,
                    F.tv_loss
                ],
                feed_dict={
                    F.input_image: batch_images,
                    F.age: batch_label_age,
                    F.gender: batch_label_gender,
                    F.z_prior: batch_z_prior
                }
            )

            print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f" %
                  (epoch + 1, num_epochs, ind_batch + 1, num_batches, EG_err, TV))
            print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
            print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

            # estimate left run time
            elapse = time.time() - start_time
            time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
            print("\tTime left: %02d:%02d:%02d" %
                  (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

            # add to summary
            summary = F.summary.eval(
                feed_dict={
                    F.input_image: batch_images,
                    F.age: batch_label_age,
                    F.gender: batch_label_gender,
                    F.z_prior: batch_z_prior
                }
            )
            F.writer.add_summary(summary, F.EG_global_step.eval())

        # save sample images for each epoch
        name = '{:02d}.png'.format(epoch + 1)
        F.sample(sample_images, sample_label_age, sample_label_gender, name)
        F.test(sample_images, sample_label_gender, name)

        # save checkpoint for each 10 epoch
        if np.mod(epoch, 10) == 9:
            F.save_checkpoint()

    # save the trained model
    F.save_checkpoint()
    # close the summary writer
    F.writer.close()